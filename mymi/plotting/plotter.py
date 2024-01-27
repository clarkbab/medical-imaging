import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap, rgb2hex
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from numpy import ndarray
import os
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
import seaborn as sns
from statannotations.Annotator import Annotator
import torchio
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from mymi import dataset
from mymi.geometry import get_box, get_extent, get_extent_centre
from mymi import logging
from mymi.postprocessing import largest_cc_3D
from mymi.regions import get_region_patch_size, is_region
from mymi.regions import truncate_spine as truncate
from mymi.transforms import crop_or_pad_box, crop_point, crop_2D, register_image
from mymi.types import Axis, Box2D, Box3D, Crop2D, Extrema, ImageSpacing2D, ImageSize2D, ImageSize3D, ImageSpacing3D, PatientID, Point3D
from mymi.utils import arg_to_list

DEFAULT_FONT_SIZE = 8

def __plot_region_data(
    data: Dict[str, np.ndarray],
    slice_idx: int,
    alpha: float,
    aspect: float,
    latex: bool,
    perimeter: bool,
    view: Axis,
    ax = None,
    cca: bool = False, connected_extent: bool = False,
    colour: Optional[Union[str, List[str]]] = None,
    crop: Optional[Box2D] = None,
    linestyle: str = 'solid',
    legend_show_all_regions: bool = False,
    show_extent: bool = False) -> bool:
    __assert_view(view)

    regions = list(data.keys()) 
    if colour is None:
        colours = sns.color_palette('colorblind', n_colors=len(regions))
    else:
        colours = arg_to_list(colour, (str, tuple))

    if not ax:
        ax = plt.gca()

    # Plot each region.
    show_legend = False
    for region, colour in zip(regions, colours):
        # Define cmap.
        cmap = ListedColormap(((1, 1, 1, 0), colour))

        # Convert data to 'imshow' co-ordinate system.
        slice_data = __get_slice_data(data[region], slice_idx, view)

        # Crop image.
        if crop:
            slice_data = crop_2D(slice_data, __reverse_box_coords_2D(crop))

        # Plot extent.
        if show_extent:
            extent = get_extent(data[region])
            label = f'{region} extent' if __box_in_plane(extent, view, slice_idx) else f'{region} extent (offscreen)'
            __plot_box_slice(extent, view, colour=colour, crop=crop, label=label, linestyle='dashed')
            show_legend = True

        # Plot connected extent.
        if connected_extent:
            extent = get_extent(largest_cc_3D(data[region]))
            if __box_in_plane(extent, view, slice_idx):
                __plot_box_slice(extent, view, colour='b', crop=crop, label=f'{region} conn. extent', linestyle='dashed')

        # Skip region if not present on this slice.
        if not legend_show_all_regions and slice_data.max() == 0:
            continue
        else:
            show_legend = True

        # Get largest component.
        if cca:
            slice_data = largest_cc_3D(slice_data)

        # Plot region.
        ax.imshow(slice_data, alpha=alpha, aspect=aspect, cmap=cmap, interpolation='none', origin=__get_origin(view))
        label = __escape_latex(region) if latex else region
        ax.plot(0, 0, c=colour, label=label)
        if perimeter:
            ax.contour(slice_data, colors=[colour], levels=[.5], linestyles=linestyle)

        # # Set ticks.
        # if crop is not None:
        #     min, max = crop
        #     width = tuple(np.array(max) - min)
        #     xticks = np.linspace(0, 10 * np.floor(width[0] / 10), 5).astype(int)
        #     xtick_labels = xticks + min[0]
        #     ax.set_xticks(xticks)
        #     ax.set_xticklabels(xtick_labels)
        #     yticks = np.linspace(0, 10 * np.floor(width[1] / 10), 5).astype(int)
        #     ytick_labels = yticks + min[1]
        #     ax.set_yticks(yticks)
        #     ax.set_yticklabels(ytick_labels)

    return show_legend

def __to_image_coords(data: ndarray) -> ndarray:
    # 'plt.imshow' expects (y, x).
    data = np.transpose(data)
    return data

def __get_origin(view: Axis) -> str:
    if view == 2:
        return 'upper'
    else:
        return 'lower'
    
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

def __get_slice_data(
    data: np.ndarray,
    slice_idx: int,
    view: Axis) -> np.ndarray:
    # Check that slice index isn't too large.
    if slice_idx >= data.shape[view]:
        raise ValueError(f"Slice '{slice_idx}' out of bounds, only '{data.shape[view]}' {__view_to_text(view)} slices.")

    # Get correct plane.
    data_index = (
        slice_idx if view == 0 else slice(data.shape[0]),
        slice_idx if view == 1 else slice(data.shape[1]),
        slice_idx if view == 2 else slice(data.shape[2]),
    )
    slice_data = data[data_index]

    # Convert from our co-ordinate system (frontal, sagittal, longitudinal) to 
    # that required by 'imshow'.
    slice_data = __to_image_coords(slice_data)

    return slice_data

def __get_aspect_ratio(
    view: Axis,
    spacing: ImageSpacing3D) -> float:
    if view == 0:
        aspect = spacing[2] / spacing[1]
    elif view == 1:
        aspect = spacing[2] / spacing[0]
    elif view == 2:
        aspect = spacing[1] / spacing[0]
    return aspect

def __reverse_box_coords_2D(box: Box2D) -> Box2D:
    # Swap x/y coordinates.
    return tuple((y, x) for x, y in box)

def __box_in_plane(
    box: Box3D,
    view: Axis,
    slice_idx: int) -> bool:
    # Get view bounding box.
    min, max = box
    min = min[view]
    max = max[view]

    # Calculate if the box is in plane.
    result = slice_idx >= min and slice_idx <= max
    return result

def __plot_box_slice(
    box: Box3D,
    view: Axis,
    colour: str = 'r',
    crop: Box2D = None,
    label: str = 'box',
    linestyle: str = 'solid') -> None:
    # Compress box to 2D.
    if view == 0:
        dims = (1, 2)
    elif view == 1:
        dims = (0, 2)
    elif view == 2:
        dims = (0, 1)
    min, max = box
    min = tuple(np.array(min)[[*dims]])
    max = tuple(np.array(max)[[*dims]])
    box_2D = (min, max)

    # Apply crop.
    if crop:
        box_2D = crop_or_pad_box(box_2D, crop)

        if box_2D is None:
            # Box has been cropped off screen.
            return

        # Reduce resulting box max by 1 to avoid plotting box outside of image.
        # This results from our treatment of box max as being 'exclusive', in line
        # with other python objects such as ranges.
        min, max = box_2D
        max = tuple(np.array(max) - 1)
        box_2D = (min, max)

    # Draw bounding box.
    min, max = box_2D
    min = np.array(min) - .5
    max = np.array(max) + .5
    width = np.array(max) - min
    rect = Rectangle(min, *width, linewidth=1, edgecolor=colour, facecolor='none', linestyle=linestyle)
    ax = plt.gca()
    ax.add_patch(rect)
    plt.plot(0, 0, c=colour, label=label, linestyle=linestyle)

def __escape_latex(text: str) -> str:
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

def __assert_data(
    ct_data: Optional[np.ndarray],
    region_data: Optional[Dict[str, np.ndarray]]):
    if ct_data is None and region_data is None:
        raise ValueError(f"Either 'ct_data' or 'region_data' must be set.")

def __assert_slice_idx(
    centre_of: Optional[int],
    extent_of: Optional[Tuple[str, bool]],
    slice_idx: Optional[str]):
    if centre_of is None and extent_of is None and slice_idx is None:
        raise ValueError(f"Either 'centre_of', 'extent_of' or 'slice_idx' must be set.")
    elif (centre_of is not None and extent_of is not None) or (centre_of is not None and slice_idx is not None) or (extent_of is not None and slice_idx is not None):
        raise ValueError(f"Only one of 'centre_of', 'extent_of' or 'slice_idx' can be set.")

def __assert_view(view: int):
    assert view in (0, 1, 2)

def plot_heatmap(
    id: str,
    heatmap: np.ndarray,
    spacing: ImageSpacing3D,
    alpha_heatmap: float = 0.7,
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[Axes] = None,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, Crop2D]] = None,
    crop_margin: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    fontsize: int = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    linestyle_pred: str = 'solid',
    linestyle_region: str = 'solid',
    pred_data: Optional[Dict[str, np.ndarray]] = None,
    region_data: Optional[Dict[str, np.ndarray]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_legend: bool = True,
    show_pred_boundary: bool = True,
    show_region_extent: bool = True,
    slice_idx: Optional[int] = None,
    view: Axis = 0, 
    **kwargs) -> None:
    __assert_slice_idx(centre_of, extent_of, slice_idx)

    # Create plot figure/axis.
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)
    else:
        show = False

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    if centre_of is not None:
        # Get 'slice_idx' at centre of data.
        label = region_data[centre_of] if type(centre_of) == str else centre_of
        extent_centre = get_extent_centre(label)
        slice_idx = extent_centre[view]

    if extent_of is not None:
        # Get 'slice_idx' at min/max extent of data.
        label = region_data[extent_of[0]] if type(extent_of[0]) == str else extent_of
        extent_end = 0 if extent_of[1] == 'min' else 1
        extent = get_extent(label)
        slice_idx = extent[extent_end][view]

    # Plot patient regions.
    size = heatmap.shape
    plot_region(id, size, spacing, alpha_region=alpha_region, aspect=aspect, ax=ax, crop=crop, crop_margin=crop_margin, ct_data=ct_data, latex=latex, legend_loc=legend_loc, linestyle_region=linestyle_region, region_data=region_data, show=False, show_extent=show_region_extent, show_legend=False, slice_idx=slice_idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'Box2D' type.
        if type(crop) == str:
            crop = __get_region_crop(region_data[crop], crop_margin, spacing, view)     # Crop was 'region_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Crop2D' type.

    # Get aspect ratio.
    if not aspect:
        aspect = __get_aspect_ratio(view, spacing) 

    # Get slice data.
    heatmap_slice = __get_slice_data(heatmap, slice_idx, view)

    # Crop the image.
    if crop is not None:
        heatmap_slice = crop_2D(heatmap_slice, __reverse_box_coords_2D(crop))

    # Plot heatmap
    ax.imshow(heatmap_slice, alpha=alpha_heatmap, aspect=aspect, origin=__get_origin(view))

    # Plot predictions.
    if pred_data is not None:
        for pred_label, pred in pred_data.items():
            if pred.sum() != 0:
                # Get slice data.
                pred_slice_data = __get_slice_data(pred, slice_idx, view)

                # Crop the image.
                if crop:
                    pred_slice_data = crop_2D(pred_slice_data, __reverse_box_coords_2D(crop))

                # Plot prediction.
                if pred_slice_data.sum() != 0: 
                    cmap = ListedColormap(((1, 1, 1, 0), colour))
                    ax.imshow(pred_slice_data, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_origin(view))
                    ax.plot(0, 0, c=colour, label=pred_label)
                    if show_pred_boundary:
                        ax.contour(pred_slice_data, colors=[colour], levels=[.5], linestyles=linestyle_pred)

            # Plot prediction extent.
            if pred.sum() != 0 and show_pred_extent:
                # Get prediction extent.
                pred_extent = get_extent(pred)

                # Plot if extent box is in view.
                label = f'{model_name} extent' if __box_in_plane(pred_extent, view, slice_idx) else f'{model_name} extent (offscreen)'
                __plot_box_slice(pred_extent, view, colour=colour, crop=crop, label=label, linestyle='dashed')

    # Show legend.
    if show_legend:
        plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

    # Revert latex settings.
    if latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })

def plot_region(
    id: str,
    size: ImageSize3D,
    spacing: ImageSpacing3D,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[Axes] = None,
    cca: bool = False,
    centre_of: Optional[Union[str, np.ndarray]] = None,             # Uses 'region_data' if 'str', else uses 'np.ndarray'.
    colour: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[str, np.ndarray, Crop2D]] = None,    # Uses 'region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    crop_centre_mm: Optional[Tuple[float, float]] = None,
    crop_margin: float = 100,                                       # Applied if cropping to 'region_data' or 'np.ndarray'.
    ct_data: Optional[np.ndarray] = None,
    window: Optional[Union[Literal['bone', 'lung', 'tissue'], Tuple[float, float]]] = 'tissue',
    dose_alpha: float = 0.3,
    dose_data: Optional[np.ndarray] = None,
    dose_legend_size: float = 0.03,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'region_data' if 'str', else 'np.ndarray') and min/max of extent.
    figsize: Tuple[int, int] = (8, 8),
    fontsize: int = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_show_all_regions: bool = False,
    linestyle_region: bool = 'solid',
    norm: Optional[Tuple[float, float]] = None,
    perimeter: bool = True,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    region_data: Optional[Dict[str, np.ndarray]] = None,            # All data passed to 'region_data' is plotted.
    savepath: Optional[str] = None,
    show: bool = True,
    show_axis: bool = True,
    show_extent: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    show_title_slice: bool = True,
    show_title_view: bool = True,
    show_x_label: bool = True,
    show_x_ticks: bool = True,
    show_y_label: bool = True,
    show_y_ticks: bool = True,
    slice_idx: Optional[int] = None,
    title: Optional[str] = None,
    transform: torchio.transforms.Transform = None,
    view: Axis = 0,
    **kwargs) -> None:
    __assert_slice_idx(centre_of, extent_of, slice_idx)
    assert view in (0, 1, 2)

    if ax is None:
        # Create figure/axes.
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)
        close_figure = True
    else:
        # Assume that parent routine will call 'plt.show()' after
        # all axes are plotted.
        show = False
        close_figure = False

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    if centre_of is not None:
        # Get 'slice_idx' at centre of data.
        label = region_data[centre_of] if type(centre_of) == str else centre_of
        if postproc:
            label = postproc(label)
        if label.sum() == 0:
            raise ValueError(f"'centre_of' (np.ndarray/region) is empty.")
        extent_centre = get_extent_centre(label)
        slice_idx = extent_centre[view]

    if extent_of is not None:
        if len(extent_of) == 2:
            eo_region, eo_end = extent_of
            eo_axis = view
        elif len(extent_of) == 3:
            eo_region, eo_end, eo_axis = extent_of

        # Get 'slice_idx' at min/max extent of data.
        label = region_data[eo_region] if type(eo_region) == str else eo_region     # 'eo_region' can be str ('region_data' key) or np.ndarray.
        assert eo_end in ('min', 'max'), "'extent_of' must have one of ('min', 'max') as second element."
        eo_end = 0 if eo_end == 'min' else 1
        if postproc:
            label = postproc(label)
        extent_voxel = get_extent_voxel(label, eo_axis, eo_end, view)
        slice_idx = extent[eo_end][axis]

    # Convert 'crop' to 'Box2D' type.
    if crop is not None:
        if isinstance(crop, str):
            crop = __get_region_crop(region_data[crop], crop_margin, spacing, view)     # Crop was 'region_data' key.
        elif isinstance(crop, np.ndarray):
            crop = __get_region_crop(crop, crop_margin, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Crop2D' type.

    if crop_centre_mm is not None:
        view_size = __get_view_size(view, size)
        view_spacing = __get_view_spacing(view, spacing)
        crop_centre = tuple((np.array(crop_centre_mm) / np.array(view_spacing)).astype(np.int32))
        logging.info(f"centre_crop={crop_centre}")
        crop = ((view_size[0] // 2 - crop_centre[0] // 2, view_size[1] // 2 - crop_centre[1] // 2), (view_size[0] // 2 + crop_centre[0] // 2, view_size[1] // 2 + crop_centre[1] // 2))
        logging.info(f"crop={crop}")

    # Apply postprocessing.
    if region_data is not None:
        if postproc:
            region_data = dict(((r, postproc(d)) for r, d in region_data.items()))

    if ct_data is not None:
        # Perform any normalisation.
        if norm is not None:
            mean, std_dev = norm
            
        # Load CT slice.
        ct_slice_data = __get_slice_data(ct_data, slice_idx, view)
        if dose_data is not None:
            dose_slice_data = __get_slice_data(dose_data, slice_idx, view)
    else:
        # Load empty slice.
        ct_slice_data = __get_slice_data(np.zeros(shape=size), slice_idx, view)

    # Perform crop on CT data or placeholder.
    if crop is not None:
        ct_slice_data = crop_2D(ct_slice_data, __reverse_box_coords_2D(crop))

        if dose_data is not None:
            dose_slice_data = crop_2D(dose_slice_data, __reverse_box_coords_2D(crop))

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = __get_aspect_ratio(view, spacing) 

    # Determine CT window.
    if ct_data is not None:
        if window is not None:
            if type(window) == str:
                if window == 'bone':
                    width, level = (2000, 300)
                elif window == 'lung':
                    width, level = (2000, -200)
                elif window == 'tissue':
                    width, level = (350, 50)
                else:
                    raise ValueError(f"Window '{window}' not recognised.")
            else:
                width, level = window
            vmin = level - (width / 2)
            vmax = level + (width / 2)
        else:
            vmin = ct_data.min()
            vmax = ct_data.max()
    else:
        vmin = 0
        vmax = 0

    # Plot CT data.
    ax.imshow(ct_slice_data, cmap='gray', aspect=aspect, interpolation='none', origin=__get_origin(view), vmin=vmin, vmax=vmax)

    if not show_axis:
        ax.set_axis_off()

    if show_x_label:
        # Add 'x-axis' label.
        if view == 0:
            spacing_x = spacing[1]
        elif view == 1: 
            spacing_x = spacing[0]
        elif view == 2:
            spacing_x = spacing[0]

        ax.set_xlabel(f'voxel [@ {spacing_x:.3f} mm spacing]')

    if show_y_label:
        # Add 'y-axis' label.
        if view == 0:
            spacing_y = spacing[2]
        elif view == 1:
            spacing_y = spacing[2]
        elif view == 2:
            spacing_y = spacing[1]
        ax.set_ylabel(f'voxel [@ {spacing_y:.3f} mm spacing]')

    if region_data is not None:
        # Plot regions.
        should_show_legend = __plot_region_data(region_data, slice_idx, alpha_region, aspect, latex, perimeter, view, ax=ax, cca=cca, colour=colour, crop=crop, legend_show_all_regions=legend_show_all_regions, linestyle=linestyle_region, show_extent=show_extent)

        # Create legend.
        if show_legend and should_show_legend:
            plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
            for l in plt_legend.get_lines():
                l.set_linewidth(8)

    # Plot dose data.
    if dose_data is not None:
        axim = ax.imshow(dose_slice_data, alpha=dose_alpha, aspect=aspect, origin=__get_origin(view))
        cbar = plt.colorbar(axim, fraction=dose_legend_size)
        cbar.set_label(label='Dose [Gray]', size=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    # Show axis markers.
    if not show_x_ticks:
        ax.get_xaxis().set_ticks([])
    if not show_y_ticks:
        ax.get_yaxis().set_ticks([])

    if show_title:
        # Add title.
        if title is None:
            # Set default title.
            n_slices = size[view]
            title = f"patient: {id}"
            if show_title_slice:
                title = f"{title}, slice: {slice_idx}/{n_slices - 1}"
            if show_title_view:
                title = f"{title} ({__view_to_text(view)} view)"

        # Escape text if using latex.
        if latex:
            title = __escape_latex(title)

        ax.set_title(title)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

        # Revert latex settings.
        if latex:
            plt.rcParams.update({
                "font.family": rc_params['font.family'],
                'text.usetex': rc_params['text.usetex']
            })

    if close_figure:
        plt.close() 

def plot_localiser_prediction(
    id: str,
    spacing: ImageSpacing3D, 
    pred_data: np.ndarray,
    pred_region: str,
    aspect: float = None,
    ax: Optional[Axes] = None,
    centre_of: Optional[str] = None,
    crop: Box2D = None,
    crop_margin: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[Extrema] = None,
    figsize: Tuple[int, int] = (8, 8),
    fontsize: float = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    linestyle_pred: str = 'solid',
    alpha_pred: float = 0.5,
    pred_centre_colour: str = 'deepskyblue',
    pred_colour: str = 'deepskyblue',
    pred_extent_colour: str = 'deepskyblue',
    region_data: Optional[Dict[str, np.ndarray]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_label_extent: bool = True,
    show_legend: bool = True,
    show_pred_centre: bool = True,
    show_pred_boundary: bool = True,
    show_pred_extent: bool = True,
    show_pred: bool = True,
    show_seg_patch: bool = True,
    slice_idx: Optional[int] = None,
    truncate_spine: bool = True,
    view: Axis = 0,
    **kwargs: dict) -> None:
    __assert_slice_idx(centre_of, extent_of, slice_idx)
    __assert_view(view)

    if ax is None:
        # Create figure/axes.
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)
        close_figure = True
    else:
        # Assume that parent routine will call 'plt.show()' after
        # all axes are plotted.
        show = False
        close_figure = False

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Load localiser segmentation.
    if pred_data.sum() == 0:
        logging.info('Empty prediction')
        empty_pred = True
    else:
        empty_pred = False

    if centre_of is not None:
        # Get 'slice_idx' at centre of data.
        label = region_data[centre_of] if type(centre_of) == str else centre_of
        extent_centre = get_extent_centre(label)
        slice_idx = extent_centre[view]

    if extent_of is not None:
        if len(extent_of) == 2:
            eo_region, eo_end = extent_of
            eo_axis = view
        elif len(extent_of) == 3:
            eo_region, eo_end, eo_axis = extent_of

        # Get 'slice_idx' at min/max extent of data.
        label = region_data[eo_region] if type(eo_region) == str else eo_region     # 'eo_region' can be str ('region_data' key) or np.ndarray.
        assert eo_end in ('min', 'max'), "'extent_of' must have one of ('min', 'max') as second element."
        eo_end = 0 if eo_end == 'min' else 1
        if postproc:
            label = postproc(label)
        extent = get_extent(label)
        slice_idx = extent[eo_end][axis]

    # Plot patient regions.
    plot_region(id, pred_data.shape, spacing, aspect=aspect, ax=ax, crop=crop, ct_data=ct_data, figsize=figsize, latex=latex, legend_loc=legend_loc, region_data=region_data, show_legend=show_legend, show_extent=show_label_extent, slice_idx=slice_idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'Box2D' type.
        if type(crop) == str:
            crop = __get_region_crop(region_data[crop], crop_margin, spacing, view)     # Crop was 'region_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Crop2D' type.

    # Plot prediction.
    if show_pred and not empty_pred:
        # Get aspect ratio.
        if not aspect:
            aspect = __get_aspect_ratio(view, spacing) 

        # Get slice data.
        pred_slice_data = __get_slice_data(pred_data, slice_idx, view)

        # Crop the image.
        if crop:
            pred_slice_data = crop_2D(pred_slice_data, __reverse_box_coords_2D(crop))

        # Plot prediction.
        colours = [(1, 1, 1, 0), pred_colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice_data, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_origin(view))
        plt.plot(0, 0, c=pred_colour, label='Loc. Prediction')
        if show_pred_boundary:
            plt.contour(pred_slice_data, colors=[pred_colour], levels=[.5], linestyles=linestyle_pred)

    # Plot prediction extent.
    if show_pred_extent and not empty_pred:
        # Get extent of prediction.
        pred_extent = get_extent(pred_data)

        # Plot extent if in view.
        label = 'Loc. extent' if __box_in_plane(pred_extent, view, slice_idx) else 'Loc. extent (offscreen)'
        __plot_box_slice(pred_extent, view, colour=pred_extent_colour, crop=crop, label=label, linestyle='dashed')

    # Plot localiser centre.
    if show_pred_centre and not empty_pred:
        # Truncate if necessary to show true pred centre.
        centre_data = truncate(pred_data, spacing) if truncate_spine and pred_region == 'SpinalCord' else pred_data

        # Get pred centre.
        pred_centre = get_extent_centre(centre_data) 

        # Get 2D loc centre.
        if view == 0:
            pred_centre = (pred_centre[1], pred_centre[2])
        elif view == 1:
            pred_centre = (pred_centre[0], pred_centre[2])
        elif view == 2:
            pred_centre = (pred_centre[0], pred_centre[1])
            
        # Apply crop.
        if crop:
            pred_centre = crop_point(pred_centre, crop)

        # Plot the prediction centre.
        if pred_centre is not None:
            plt.scatter(*pred_centre, c=pred_centre_colour, label=f"Loc. Centre")
        else:
            plt.plot(0, 0, c=pred_centre_colour, label='Loc. Centre (offscreen)')

    # Plot second stage patch.
    if not empty_pred and show_seg_patch:
        # Truncate if necessary to show true pred centre.
        centre_data = truncate(pred_data, spacing) if truncate_spine and pred_region == 'SpinalCord' else pred_data

        # Get pred centre.
        pred_centre = get_extent_centre(centre_data) 

        # Get second-stage patch.
        size = get_region_patch_size(pred_region, spacing)
        min, max = get_box(pred_centre, size)

        # Squash min/max to label size.
        min = np.clip(min, a_min=0, a_max=None)
        max = np.clip(max, a_min=None, a_max=pred_data.shape)

        if __box_in_plane((min, max), view, slice_idx):
            __plot_box_slice((min, max), view, colour='tomato', crop=crop, label='Seg. Patch', linestyle='dotted')
        else:
            plt.plot(0, 0, c='tomato', label='Seg. Patch (offscreen)', linestyle='dashed')

    # Show legend.
    if show_legend:
        plt_legend = plt.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.axes(frameon=False)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

        # Revert latex settings.
        if latex:
            plt.rcParams.update({
                "font.family": rc_params['font.family'],
                'text.usetex': rc_params['text.usetex']
            })

    if close_figure:
        plt.close() 

def __get_region_crop(
    data: np.ndarray,
    crop_margin: float,
    spacing: ImageSpacing3D,
    view: Axis) -> Box2D:
    # Get 3D crop box.
    extent = get_extent(data)

    # Add crop margin.
    crop_margin_vox = tuple(np.ceil(np.array(crop_margin) / spacing).astype(int))
    min, max = extent
    min = tuple(np.array(min) - crop_margin_vox)
    max = tuple(np.array(max) + crop_margin_vox)

    # Don't pad original image.
    min = tuple(np.clip(min, a_min=0, a_max=None))
    max = tuple(np.clip(max, a_min=None, a_max=data.shape))

    # Select 2D component.
    if view == 0:
        min = (min[1], min[2])
        max = (max[1], max[2])
    elif view == 1:
        min = (min[0], min[2])
        max = (max[0], max[2])
    elif view == 2:
        min = (min[0], min[1])
        max = (max[0], max[1])
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
    n_bins = int(np.ceil((max - min) / resolution))

    # Get limits.
    if range:
        limits = range
    else:
        limits = (min, max)
        
    # Plot histogram.
    plt.figure(figsize=figsize)
    plt.hist(data.flatten(), bins=n_bins, range=range, histtype='step',edgecolor='r',linewidth=3)
    plt.title(f'Hist. of voxel values, range={tuple(np.array(limits).round().astype(int))}')
    plt.xlabel('HU')
    plt.ylabel('Frequency')
    plt.show()

def plot_multi_segmenter_prediction(
    id: str,
    spacing: ImageSpacing3D,
    pred_data: Dict[str, np.ndarray],
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: float = None,
    ax: Optional[Axes] = None,
    centre_of: Optional[str] = None,
    colour: Optional[Union[str, List[str]]] = None,
    colour_match: bool = False,
    crop: Optional[Union[str, Box2D]] = None,
    crop_margin: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[float, float] = (8, 8),
    fontsize: float = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    linestyle_pred: str = 'solid',
    linestyle_region: str = 'solid',
    region_data: Optional[Dict[str, np.ndarray]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_legend: bool = True,
    show_pred: bool = True,
    show_pred_boundary: bool = True,
    show_pred_extent: bool = True,
    show_region_extent: bool = True,
    slice_idx: Optional[int] = None,
    view: Axis = 0,
    **kwargs: dict) -> None:
    __assert_slice_idx(centre_of, extent_of, slice_idx)
    model_names = tuple(pred_data.keys())
    n_models = len(model_names)
    n_regions = len(region_data.keys()) if region_data is not None else 0

    # Create plot figure/axis.
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)

    # Get unique colours.
    if colour is None:
        if colour_match:
            n_colours = np.max((n_regions, n_models))
        else:
            n_colours = n_regions + n_models
        colours = sns.color_palette('colorblind', n_colours)
    else:
        colours = arg_to_list(colour, (str, tuple))

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Print prediction summary info.
    for model_name, pred in pred_data.items():
        logging.info(f"""
Prediction: {model_name}""")
        if pred.sum() != 0:
            volume_vox = pred.sum()
            volume_mm3 = volume_vox * np.product(spacing)
            logging.info(f"""
    Volume (vox): {volume_vox}
    Volume (mm^3): {volume_mm3:.2f}""")
        else:
            logging.info(f"""
    Empty""")

    if centre_of is not None:
        # Get 'slice_idx' at centre of data.
        label = region_data[centre_of] if type(centre_of) == str else centre_of
        extent_centre = get_extent_centre(label)
        slice_idx = extent_centre[view]

    if extent_of is not None:
        # Get 'slice_idx' at min/max extent of data.
        label = region_data[extent_of[0]] if type(extent_of[0]) == str else extent_of
        extent_end = 0 if extent_of[1] == 'min' else 1
        extent = get_extent(label)
        slice_idx = extent[extent_end][view]

    # Plot patient regions - even if no 'ct_data/region_data' we still want to plot shape as black background.
    size = ct_data.shape
    region_colours = colours if colour_match else colours[:n_regions]
    plot_region(id, size, spacing, alpha_region=alpha_region, aspect=aspect, ax=ax, colour=region_colours, crop=crop, crop_margin=crop_margin, ct_data=ct_data, latex=latex, legend_loc=legend_loc, linestyle_region=linestyle_region, region_data=region_data, show=False, show_extent=show_region_extent, show_legend=False, slice_idx=slice_idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'Box2D' type.
        if type(crop) == str:
            crop = __get_region_crop(region_data[crop], crop_margin, spacing, view)     # Crop was 'region_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Crop2D' type.

    # Plot predictions.
    for i in range(n_models):
        model_name = model_names[i]
        pred = pred_data[model_name]
        colour = colours[i] if colour_match else colours[n_regions + i]

        if pred.sum() != 0 and show_pred:
            # Get aspect ratio.
            if not aspect:
                aspect = __get_aspect_ratio(view, spacing) 

            # Get slice data.
            pred_slice_data = __get_slice_data(pred, slice_idx, view)

            # Crop the image.
            if crop:
                pred_slice_data = crop_2D(pred_slice_data, __reverse_box_coords_2D(crop))

            # Plot prediction.
            if pred_slice_data.sum() != 0: 
                cmap = ListedColormap(((1, 1, 1, 0), colour))
                ax.imshow(pred_slice_data, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_origin(view))
                ax.plot(0, 0, c=colour, label=model_name)
                if show_pred_boundary:
                    ax.contour(pred_slice_data, colors=[colour], levels=[.5], linestyles=linestyle_pred)

        # Plot prediction extent.
        if pred.sum() != 0 and show_pred_extent:
            # Get prediction extent.
            pred_extent = get_extent(pred)

            # Plot if extent box is in view.
            label = f'{model_name} extent' if __box_in_plane(pred_extent, view, slice_idx) else f'{model_name} extent (offscreen)'
            __plot_box_slice(pred_extent, view, colour=colour, crop=crop, label=label, linestyle='dashed')

    # Show legend.
    if show_legend:
        plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

    # Revert latex settings.
    if latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })

def plot_segmenter_prediction(
    id: str,
    spacing: ImageSpacing3D,
    pred_data: Dict[str, np.ndarray],
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: float = None,
    ax: Optional[Axes] = None,
    centre_of: Optional[str] = None,
    colour: Optional[Union[str, List[str]]] = None,
    colour_match: bool = False,
    crop: Optional[Union[str, Box2D]] = None,
    crop_margin: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[float, float] = (8, 8),
    fontsize: float = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    linestyle_pred: str = 'solid',
    linestyle_region: str = 'solid',
    loc_centre: Optional[Union[Point3D, List[Point3D]]] = None,
    region_data: Optional[Dict[str, np.ndarray]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_legend: bool = True,
    show_loc_centre: bool = True,
    show_pred: bool = True,
    show_pred_boundary: bool = True,
    show_pred_extent: bool = True,
    show_pred_patch: bool = False,
    show_region_extent: bool = True,
    slice_idx: Optional[int] = None,
    view: Axis = 0,
    **kwargs: dict) -> None:
    __assert_slice_idx(centre_of, extent_of, slice_idx)
    model_names = tuple(pred_data.keys())
    n_models = len(model_names)
    n_regions = len(region_data.keys()) if region_data is not None else 0
    loc_centres = arg_to_list(loc_centre, tuple)
    if loc_centres is not None:
        assert len(loc_centres) == n_models

    # Create plot figure/axis.
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)

    # Get unique colours.
    if colour is None:
        if colour_match:
            n_colours = np.max((n_regions, n_models))
        else:
            n_colours = n_regions + n_models
        colours = sns.color_palette('colorblind', n_colours)
    else:
        colours = arg_to_list(colour, [str, tuple])

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Print prediction summary info.
    for model_name, pred in pred_data.items():
        logging.info(f"""
Prediction: {model_name}""")
        if pred.sum() != 0:
            volume_vox = pred.sum()
            volume_mm3 = volume_vox * np.product(spacing)
            logging.info(f"""
    Volume (vox): {volume_vox}
    Volume (mm^3): {volume_mm3:.2f}""")
        else:
            logging.info(f"""
    Empty""")

    if centre_of is not None:
        # Get 'slice_idx' at centre of data.
        label = region_data[centre_of] if type(centre_of) == str else centre_of
        extent_centre = get_extent_centre(label)
        slice_idx = extent_centre[view]

    if extent_of is not None:
        # Get 'slice_idx' at min/max extent of data.
        label = region_data[extent_of[0]] if type(extent_of[0]) == str else extent_of
        extent_end = 0 if extent_of[1] == 'min' else 1
        extent = get_extent(label)
        slice_idx = extent[extent_end][view]

    # Plot patient regions - even if no 'ct_data/region_data' we still want to plot shape as black background.
    size = ct_data.shape
    region_colours = colours if colour_match else colours[:n_regions]
    plot_region(id, size, spacing, alpha_region=alpha_region, aspect=aspect, ax=ax, colour=region_colours, crop=crop, crop_margin=crop_margin, ct_data=ct_data, latex=latex, legend_loc=legend_loc, linestyle_region=linestyle_region, region_data=region_data, show=False, show_extent=show_region_extent, show_legend=False, slice_idx=slice_idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'Box2D' type.
        if type(crop) == str:
            crop = __get_region_crop(region_data[crop], crop_margin, spacing, view)     # Crop was 'region_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Crop2D' type.

    # Plot predictions.
    for i in range(n_models):
        model_name = model_names[i]
        pred = pred_data[model_name]
        colour = colours[i] if colour_match else colours[n_regions + i]
        loc_centre = loc_centres[i]

        if pred.sum() != 0 and show_pred:
            # Get aspect ratio.
            if not aspect:
                aspect = __get_aspect_ratio(view, spacing) 

            # Get slice data.
            pred_slice_data = __get_slice_data(pred, slice_idx, view)

            # Crop the image.
            if crop:
                pred_slice_data = crop_2D(pred_slice_data, __reverse_box_coords_2D(crop))

            # Plot prediction.
            if pred_slice_data.sum() != 0: 
                cmap = ListedColormap(((1, 1, 1, 0), colour))
                ax.imshow(pred_slice_data, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_origin(view))
                ax.plot(0, 0, c=colour, label=model_name)
                if show_pred_boundary:
                    ax.contour(pred_slice_data, colors=[colour], levels=[.5], linestyles=linestyle_pred)

        # Plot prediction extent.
        if pred.sum() != 0 and show_pred_extent:
            # Get prediction extent.
            pred_extent = get_extent(pred)

            # Plot if extent box is in view.
            label = f'{model_name} extent' if __box_in_plane(pred_extent, view, slice_idx) else f'{model_name} extent (offscreen)'
            __plot_box_slice(pred_extent, view, colour=colour, crop=crop, label=label, linestyle='dashed')

        # Plot localiser centre.
        if show_loc_centre:
            # Get loc centre.
            loc_centre = loc_centres[i]
            if view == 0:
                centre = (loc_centre[1], loc_centre[2])
            elif view == 1:
                centre = (loc_centre[0], loc_centre[2])
            elif view == 2:
                centre = (loc_centre[0], loc_centre[1])
                
            # Apply crop.
            if crop:
                centre = crop_point(centre, crop)

            if centre:
                ax.scatter(*centre, c='royalblue', label=f"Loc. Centre")
            else:
                ax.plot(0, 0, c='royalblue', label='Loc. Centre (offscreen)')

        # Plot second stage patch.
        if loc_centre is not None and pred.sum() != 0 and show_pred_patch:
            # Get 3D patch - cropped to label size.
            region = segmenter[i][0].split('-')[1]
            size = get_region_patch_size(region, spacing)
            patch = get_box(loc_centre, size)
            label_box = ((0, 0, 0), prediction.shape)
            patch = crop_or_pad_box(patch, label_box)

            # Plot box.
            if patch and __box_in_plane(patch, view, slice_idx):
                __plot_box_slice(patch, view, colour=colour, crop=crop, label='Pred. Patch', linestyle='dotted')
            else:
                ax.plot(0, 0, c=colour, label='Pred. Patch (offscreen)', linestyle='dashed')

    # Show legend.
    if show_legend:
        plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

    # Revert latex settings.
    if latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })

def plot_segmenter_prediction_diff(
    id: str,
    spacing: ImageSpacing3D,
    diff_data: Dict[str, np.ndarray],
    alpha_diff: float = 1.0,
    aspect: float = None,
    ax: Optional[Axes] = None,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    crop_margin: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
    fontsize: float = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    linestyle_diff: str = 'solid',
    savepath: Optional[str] = None,
    show: bool = True,
    show_diff: bool = True,
    show_diff_contour: bool = False,
    show_legend: bool = True,
    slice_idx: Optional[int] = None,
    view: Axis = 0,
    **kwargs: dict) -> None:
    __assert_slice_idx(centre_of, extent_of, slice_idx)
    __assert_view(view)
    diff_names = list(diff_data.keys())
    n_diffs = len(diff_names)
    colours = sns.color_palette('husl', n_diffs)

    # Create plot figure/axis.
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Print diff summary info.
    for diff_name, diff in diff_data.items():
        logging.info(f"""
Diff: {diff_name}""")
        if diff.sum() != 0:
            volume_vox = diff.sum()
            volume_mm3 = volume_vox * np.product(spacing)
            logging.info(f"""
    Volume (vox): {volume_vox}
    Volume (mm^3): {volume_mm3:.2f}""")
        else:
            logging.info(f"""
    Empty""")

    # Plot CT data.
    size = ct_data.shape
    plot_region(id, size, spacing, aspect=aspect, ax=ax, crop=crop, crop_margin=crop_margin, ct_data=ct_data, latex=latex, legend_loc=legend_loc, show=False, show_legend=False, slice_idx=slice_idx, view=view, **kwargs)

    if crop is not None:
        crop = tuple(zip(*crop))    # Convert from 'Crop2D' to 'Box2D'.

    # Plot diffs.
    for i in range(n_diffs):
        model_name = diff_names[i]
        diff = diff_data[model_name]
        colour = colours[i]

        if diff.sum() != 0 and show_diff:
            # Get aspect ratio.
            if not aspect:
                aspect = __get_aspect_ratio(view, spacing) 

            # Get slice data.
            slice_data = __get_slice_data(diff, slice_idx, view)

            # Crop the image.
            if crop:
                slice_data = crop_2D(slice_data, __reverse_box_coords_2D(crop))

            # Plot prediction.
            cmap = ListedColormap(((1, 1, 1, 0), colour))
            ax.imshow(slice_data, alpha=alpha_diff, aspect=aspect, cmap=cmap, origin=__get_origin(view))
            ax.plot(0, 0, c=colour, label=model_name)
            if show_diff_contour:
                ax.contour(slice_data, colors=[colour], levels=[.5], linestyles=linestyle_diff)

    # Show legend.
    if show_legend:
        plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

    # Revert latex settings.
    if latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })

def plot_dataframe_v2(
    ax: Optional[Axes] = None,
    data: Optional[pd.DataFrame] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    dpi: float = 1000,
    exclude_x: Optional[Union[str, List[str]]] = None,
    figsize: Tuple[float, float] = (16, 6),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontsize_label: Optional[float] = None,
    fontsize_legend: Optional[float] = None,
    fontsize_stats: Optional[float] = None,
    fontsize_tick_label: Optional[float] = None,
    fontsize_title: Optional[float] = None,
    hue_connections_index: Optional[Union[str, List[str]]] = None,
    hue_hatch: Optional[Union[str, List[str]]] = None,
    hue_label: Optional[Union[str, List[str]]] = None,
    hue_order: Optional[List[str]] = None,
    include_x: Optional[Union[str, List[str]]] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: str = 'upper right',
    linecolour: str = 'black',
    linewidth: float = 0.5,
    major_tick_freq: Optional[float] = None,
    minor_tick_freq: Optional[float] = None,
    n_cols: Optional[int] = None,
    outlier_legend_loc: str = 'upper left',
    palette: Optional[sns.palettes._ColorPalette] = sns.color_palette('colorblind'),
    pointsize: float = 10,
    savepath: Optional[str] = None,
    share_y: bool = False,
    show_boxes: bool = True,
    show_hue_connections: bool = False,
    show_hue_connections_inliers: bool = False,
    show_legend: bool = True,
    show_points: bool = True,
    show_stats: bool = False,
    show_x_tick_labels: bool = True,
    show_x_tick_label_counts: bool = True,
    stats_exclude: List[str] = [],
    stats_exclude_left: List[str] = [],
    stats_exclude_right: List[str] = [],
    stats_index: Optional[Union[str, List[str]]] = None,
    stats_paired: bool = True,
    stats_bar_alg_use_lowest_level: bool = True,
    stats_bar_alpha: float = 0.5,
    stats_bar_data_offset: float = 0.03,
    stats_bar_grid_offset: float = 0.01,
    stats_bar_height: float = 0.01,
    stats_bar_offset: float = 0.05,
    stats_bar_text_offset: float = 0.01,
    stats_bar_show_direction: bool = False,
    stats_two_sided: bool = False,
    style: Optional[Literal['box', 'violin']] = 'box',
    ticklength: float = 0.5,
    title: Optional[str] = None,
    title_x: Optional[float] = None,
    title_y: Optional[float] = None,
    x_label: Optional[str] = None,
    x_lim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
    x_order: Optional[List[str]] = None,
    x_width: float = 0.8,
    x_tick_label: Optional[List[str]] = None,
    x_tick_label_rot: float = 0,
    y_label: Optional[str] = None,
    y_lim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None)):
    hue_hatches = arg_to_list(hue_hatch, str)
    hue_labels = arg_to_list(hue_label, str)
    include_xs = arg_to_list(include_x, str)
    exclude_xs = arg_to_list(exclude_x, str)
    if show_hue_connections and hue_connections_index is None:
        raise ValueError(f"Please set 'hue_connections_index' to allow matching points between hues.")
    if show_stats and stats_index is None:
        raise ValueError(f"Please set 'stats_index' to determine sample pairing for Wilcoxon test.")
    x_tick_labels = arg_to_list(x_tick_label, str)

    # Set default fontsizes.
    if fontsize_label is None:
        fontsize_label = fontsize
    if fontsize_legend is None:
        fontsize_legend = fontsize
    if fontsize_stats is None:
        fontsize_stats = fontsize
    if fontsize_tick_label is None:
        fontsize_tick_label = fontsize
    if fontsize_title is None:
        fontsize_title = fontsize
        
    # Include/exclude.
    if include_xs is not None:
        data = data[data[x].isin(include_xs)]
    if exclude_xs is not None:
        data = data[~data[x].isin(exclude_xs)]

    # Add outlier data.
    data = __add_outlier_info(data, x, y, hue)

    # Get min/max values for y-lim.
    if share_y:
        min_y = data[y].min()
        max_y = data[y].max()

    # Get x values.
    if x_order is None:
        x_order = list(sorted(data[x].unique()))

    # Determine x labels.
    groupby = x if hue is None else [x, hue]
    count_map = data.groupby(groupby)[y].count()
    if x_tick_labels is None:
        x_tick_labels = []
        for x_val in x_order:
            count = count_map.loc[x_val]
            if hue is not None:
                ns = count.values
                # Use a single number, e.g. (n=99) if all hues have the same number of points.
                if len(np.unique(ns)) == 1:
                    ns = ns[:1]
            else:
                ns = [count]
            label = f"{x_val}\n(n={','.join([str(n) for n in ns])})" if show_x_tick_label_counts else x_val
            x_tick_labels.append(label)

    # Create subplots if required.
    if n_cols is None:
        n_cols = len(x_order)
    n_rows = int(np.ceil(len(x_order) / n_cols))
    if ax is not None:
        assert n_rows == 1
        axs = [ax]
        # Figsize will have been handled externally.
    else:
        if n_rows > 1:
            _, axs = plt.subplots(n_rows, 1, dpi=dpi, figsize=(figsize[0], n_rows * figsize[1]), sharey=share_y)
        else:
            plt.figure(dpi=dpi, figsize=figsize)
            axs = [plt.gca()]

    # Get x-axis limits.
    x_lim = list(x_lim)
    if x_lim[0] is None:
        x_lim[0] = -0.5
    if x_lim[1] is None:
        x_lim[1] = n_cols - 0.5

    # Get hue order/colour/labels.
    if hue is not None:
        if hue_order is None:
            hue_order = list(sorted(data[hue].unique()))

        # Calculate x width for each hue.
        hue_width = x_width / len(hue_order) 

        # Check there are enough colours in palette.
        if len(hue_order) > len(palette):
            raise ValueError(f"'palette' doesn't have enough colours for hues '{hue_order}', needs '{len(hue_order)}'.")

        # Create map from hue to colour.
        hue_colours = dict((h, palette[i]) for i, h in enumerate(hue_order))

        if hue_labels is not None:
            if len(hue_labels) != len(hue_order):
                raise ValueError(f"Length of 'hue_labels' ({hue_labels}) should match hues ({hue_order}).")

    # Plot rows.
    for i in range(n_rows):
        # Split data.
        row_x_order = x_order[i * n_cols:(i + 1) * n_cols]
        row_x_tick_labels = x_tick_labels[i * n_cols:(i + 1) * n_cols]

        # Get x colours.
        if hue is None:
            # Check there are enough colors in palette.
            if len(row_x_order) > len(palette):
                raise ValueError(f"'palette' doesn't have enough colours for x values '{row_x_order}', needs '{len(row_x_order)}'.")
            x_colours = dict((x, palette[i]) for i, x in enumerate(row_x_order))

        # Get row data.
        row_df = data[data[x].isin(row_x_order)].copy()

        # Keep track of legend items.
        hue_artists = {}

        for j, x_val in enumerate(row_x_order):
            # Add x positions.
            if hue is not None:
                for k, hue_name in enumerate(hue_order):
                    x_pos = j - 0.5 * x_width + (k + 0.5) * hue_width
                    row_df.loc[(row_df[x] == x_val) & (row_df[hue] == hue_name), 'x_pos'] = x_pos
            else:
                x_pos = j
                row_df.loc[row_df[x] == x_val, 'x_pos'] = x_pos
                
            # Plot boxes.
            if show_boxes:
                if hue is not None:
                    for k, hue_name in enumerate(hue_order):
                        # Get hue data and pos.
                        hue_df = row_df[(row_df[x] == x_val) & (row_df[hue] == hue_name)]
                        if len(hue_df) == 0:
                            continue
                        hue_pos = hue_df.iloc[0]['x_pos']

                        # Get hue 'label' - allows us to use names more display-friendly than the data values.
                        hue_label = hue_name if hue_labels is None else hue_labels[k]

                        hatch = hue_hatches[k] if hue_hatches is not None else None
                        if style == 'box':
                            # Plot box.
                            res = axs[i].boxplot(hue_df[y].dropna(), boxprops=dict(color=linecolour, facecolor=hue_colours[hue_name], linewidth=linewidth), capprops=dict(color=linecolour, linewidth=linewidth), flierprops=dict(color=linecolour, linewidth=linewidth, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=linewidth), patch_artist=True, positions=[hue_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=linewidth), widths=hue_width)
                            if hatch is not None:
                                mpl.rcParams['hatch.linewidth'] = linewidth
                                res['boxes'][0].set_hatch(hatch)
                                # res['boxes'][0].set(hatch=hatch)
                                # res['boxes'][0].set_edgecolor('white')
                                # res['boxes'][0].set(facecolor='white')

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                        elif style == 'violin':
                            # Plot violin.
                            res = axs[i].violinplot(hue_df[y], positions=[hue_pos], widths=hue_width)

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                else:
                    # Plot box.
                    x_df = row_df[row_df[x] == x_val]
                    if len(x_df) == 0:
                        continue
                    x_pos = x_df.iloc[0]['x_pos']
                    if style == 'box':
                        axs[i].boxplot(x_df[y], boxprops=dict(color=linecolour, facecolor=x_colours[x_val], linewidth=linewidth), capprops=dict(color=linecolour, linewidth=linewidth), flierprops=dict(color=linecolour, linewidth=linewidth, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=linewidth), patch_artist=True, positions=[x_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=linewidth))
                    elif style == 'violin':
                        axs[i].violinplot(x_df[y], positions=[x_pos])

            # Plot points.
            if show_points:
                if hue is not None:
                    for j, hue_name in enumerate(hue_order):
                        hue_df = row_df[(row_df[x] == x_val) & (row_df[hue] == hue_name)]
                        if len(hue_df) == 0:
                            continue
                        res = axs[i].scatter(hue_df['x_pos'], hue_df[y], color=hue_colours[hue_name], edgecolors=linecolour, linewidth=linewidth, s=pointsize, zorder=100)
                        if not hue_label in hue_artists:
                            hue_artists[hue_label] = res
                else:
                    x_df = row_df[row_df[x] == x_val]
                    axs[i].scatter(x_df['x_pos'], x_df[y], color=x_colours[x_val], edgecolors=linecolour, linewidth=linewidth, s=pointsize, zorder=100)

            # Identify connections between hues.
            if hue is not None and show_hue_connections:
                # Get column/value pairs to group across hue levels.
                # line_ids = row_df[(row_df[x] == x_val) & row_df['outlier']][outlier_cols]
                x_df = row_df[(row_df[x] == x_val)]
                if not show_hue_connections_inliers:
                    line_ids = x_df[x_df['outlier']][hue_connections_index]
                else:
                    line_ids = x_df[hue_connections_index]

                # Drop duplicates.
                line_ids = line_ids.drop_duplicates()

                # Get palette.
                line_palette = sns.color_palette('husl', n_colors=len(line_ids))

                # Plot lines.
                artists = []
                labels = []
                for j, (_, line_id) in enumerate(line_ids.iterrows()):
                    # Get line data.
                    x_df = row_df[(row_df[x] == x_val)]
                    for k, v in zip(line_ids.columns, line_id):
                        x_df = x_df[x_df[k] == v]
                    x_df = x_df.sort_values('x_pos')
                    x_pos = x_df['x_pos'].tolist()
                    y_data = x_df[y].tolist()

                    # Plot line.
                    lines = axs[i].plot(x_pos, y_data, color=line_palette[j])

                    # Save line/label for legend.
                    artists.append(lines[0])
                    label = ':'.join(line_id.tolist())
                    labels.append(label)

                # Annotate outlier legend.
                if show_legend:
                    # Save main legend.
                    main_legend = axs[i].get_legend()

                    # Show outlier legend.
                    axs[i].legend(artists, labels, bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize_legend, loc=outlier_legend_loc)

                    # Re-add main legend.
                    axs[i].add_artist(main_legend)

        if hue is not None:
            if show_legend:
                # Filter 'hue_labels' based on hue 'artists'. Some hues may not be present in this row,
                # and 'hue_labels' is a global (across all rows) tracker.
                hue_labels = hue_order if hue_labels is None else hue_labels
                labels, artists = list(zip(*[(h, hue_artists[h]) for h in hue_labels if h in hue_artists]))

                # Show legend.
                legend = axs[i].legend(artists, labels, bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize_legend, loc=legend_loc)
                frame = legend.get_frame()
                frame.set_boxstyle('square')
                frame.set_edgecolor('black')
                frame.set_linewidth(linewidth)

        # Plot statistical significance.
        if hue is not None and show_stats:
            for x_val in row_x_order:
                x_df = row_df[row_df[x] == x_val]

                # Create pairs - start at lower numbers of skips as this will result in a 
                # condensed plot.
                pairs = []
                max_skips = len(hue_order) - 1
                for skip in range(1, max_skips + 1):
                    for j, hue_val in enumerate(hue_order):
                        other_hue_index = j + skip
                        if other_hue_index < len(hue_order):
                            pair = (hue_val, hue_order[other_hue_index])
                            pairs.append(pair)

                # Calculate p-values.
                # Remove pairs with no available data or no statistical significance.
                pairs_tmp = pairs.copy()
                pairs = []
                p_vals = []
                for hue_l, hue_r in pairs_tmp:
                    x_pivot_df = x_df.pivot(index=stats_index, columns=[hue], values=[y]).reset_index()
                    if (y, hue_l) in x_pivot_df.columns and (y, hue_r) in x_pivot_df.columns:
                        vals_a = x_pivot_df[y][hue_l]
                        vals_b = x_pivot_df[y][hue_r]
                        if np.any(vals_a.isna()) or np.any(vals_b.isna()):
                            raise ValueError(f"Unpaired data for x ({x}={x_val}) and hues ({hue}=({hue_l},{hue_r})).")
                        if not stats_paired:
                            vals_a = vals_a[~vals_a.isna()]
                            vals_b = vals_b[~vals_b.isna()]
                        pair = (hue_l, hue_r)
                        if stats_two_sided:
                            # Perform two-sided 'Wilcoxon signed rank test'.
                            if stats_paired:
                                _, p_val = wilcoxon(vals_a, vals_b, alternative='two-sided')
                            else:
                                _, p_val = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                            if p_val <= 0.05:
                                p_vals.append((p_val, ''))
                                pairs.append(pair)
                        else:
                            # Perform one-sided 'Wilcoxon signed rank test'.
                            if stats_paired:
                                _, p_val = wilcoxon(vals_a, vals_b, alternative='greater')
                            else:
                                _, p_val = mannwhitneyu(vals_a, vals_b, alternative='greater')
                            if p_val <= 0.05:
                                p_vals.append((p_val, '>'))
                                pairs.append(pair)
                            else:
                                if stats_paired:
                                    _, p_val = wilcoxon(vals_a, vals_b, alternative='less')
                                else:
                                    _, p_val = mannwhitneyu(vals_a, vals_b, alternative='less')
                                if p_val <= 0.05:
                                    p_vals.append((p_val, '<'))
                                    pairs.append(pair)

                # Exclude pairs.
                pairs_tmp = pairs.copy()
                p_vals_tmp = p_vals.copy()
                pairs = []
                p_vals = []
                for (hue_l, hue_r), p_val in zip(pairs_tmp, p_vals_tmp):
                    if (hue_l in stats_exclude or hue_r in stats_exclude) or (hue_l in stats_exclude_left) or (hue_r in stats_exclude_right):
                        continue
                    pairs.append((hue_l, hue_r))
                    p_vals.append(p_val)

                # Calculate grid offset.
                y_grid_offset = np.inf
                min_skip = None
                for hue_l, hue_r in pairs:
                    if stats_bar_alg_use_lowest_level:
                        skip = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                        if min_skip is None:
                            min_skip = skip
                        elif skip > min_skip:
                            continue

                    hue_l_df = x_df[x_df[hue] == hue_l]
                    hue_r_df = x_df[x_df[hue] == hue_r]
                    y_max = max(hue_l_df[y].max(), hue_r_df[y].max())
                    y_max = y_max + stats_bar_data_offset
                    if y_max < y_grid_offset:
                        y_grid_offset = y_max

                # Add grid offset.
                y_grid_offset = y_grid_offset + stats_bar_grid_offset

                # Format p-values.
                p_vals = __format_p_values(p_vals, show_direction=stats_bar_show_direction) 

                # Annotate figure.
                bar_positions = {}
                for (hue_l, hue_r), p_val in zip(pairs, p_vals):
                    # Get x positions.
                    hue_l_df = x_df[x_df[hue] == hue_l]
                    hue_r_df = x_df[x_df[hue] == hue_r]
                    x_left = hue_l_df['x_pos'].iloc[0]
                    x_right = hue_r_df['x_pos'].iloc[0]

                    # Get y grid index based on data.
                    y_maxes = [hue_df[y].max() for hue_df in [hue_l_df, hue_r_df]]
                    n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                    for j in range(n_mid_hues):
                        hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                        hue_mid_df = x_df[x_df[hue] == hue_mid]
                        y_max = hue_mid_df[y].max()
                        y_maxes.append(y_max)
                    y_max = max(y_maxes) + stats_bar_data_offset
                    y_i_data = int(np.ceil((y_max - y_grid_offset) / stats_bar_offset))

                    # Get possible y grid index based on existing bar positions.
                    ## Get existing bar positions for all 'middle' hues.
                    n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                    mid_hues_bar_positions = []
                    for j in range(n_mid_hues):
                        hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                        if hue_mid not in bar_positions:
                            continue
                        hue_mid_bar_positions = bar_positions[hue_mid]
                        mid_hues_bar_positions.append(hue_mid_bar_positions)

                    ## Find first free position greater than or equal to 'y_i_data'.
                    y_i = y_i_data
                    y_i_max = 100
                    for y_i in range(y_i_data, y_i_max):
                        if not any([y_i in p for p in mid_hues_bar_positions]):
                            break

                    # Plot bar.
                    y_min = y_grid_offset + y_i * stats_bar_offset
                    y_max = y_min + stats_bar_height
                    axs[i].plot([x_left, x_left, x_right, x_right], [y_min, y_max, y_max, y_min], alpha=stats_bar_alpha, color=linecolour, linewidth=linewidth)    

                    # Plot p-value.
                    x_text = (x_left + x_right) / 2
                    y_text = y_max + stats_bar_text_offset
                    axs[i].text(x_text, y_text, p_val, alpha=stats_bar_alpha, fontsize=fontsize_stats, horizontalalignment='center', verticalalignment='center')

                    # Track bar positions.
                    if not hue_l in bar_positions:
                        bar_positions[hue_l] = [y_i]
                    elif y_i not in bar_positions[hue_l]:
                        bar_positions[hue_l] = list(sorted(bar_positions[hue_l] + [y_i]))
                    if not hue_r in bar_positions:
                        bar_positions[hue_r] = [y_i]
                    elif y_i not in bar_positions[hue_r]:
                        bar_positions[hue_r] = list(sorted(bar_positions[hue_r] + [y_i]))
          
        # Set axis labels.
        x_label = x_label if x_label is not None else ''
        y_label = y_label if y_label is not None else ''
        axs[i].set_xlabel(x_label, fontsize=fontsize_label)
        axs[i].set_ylabel(y_label, fontsize=fontsize_label)
                
        # Set axis tick labels.
        axs[i].set_xticks(list(range(len(row_x_tick_labels))))
        if show_x_tick_labels:
            axs[i].set_xticklabels(row_x_tick_labels, fontsize=fontsize_tick_label, rotation=x_tick_label_rot)
        else:
            axs[i].set_xticklabels([])

        axs[i].tick_params(axis='y', which='major', labelsize=fontsize_tick_label)

        # Set axis limits.
        axs[i].set_xlim(*x_lim)
        y_margin = 0.05
        if not share_y:
            min_y = row_df[y].min()
            max_y = row_df[y].max()
        y_lim_row = list(y_lim)
        if y_lim_row[0] is None:
            if y_lim_row[1] is None:
                width = max_y - min_y
                y_lim_row[0] = min_y - y_margin * width
                y_lim_row[1] = max_y + y_margin * width
            else:
                width = y_lim_row[1] - min_y
                y_lim_row[0] = min_y - y_margin * width
        else:
            if y_lim_row[1] is None:
                width = max_y - y_lim_row[0]
                y_lim_row[1] = max_y + y_margin * width
        axs[i].set_ylim(*y_lim_row)

        # Set y axis major ticks.
        if major_tick_freq is not None:
            major_tick_min = y_lim[0]
            if major_tick_min is None:
                major_tick_min = axs[i].get_ylim()[0]
            major_tick_max = y_lim[1]
            if major_tick_max is None:
                major_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'major_tick_freq'.
            major_tick_min = np.ceil(major_tick_min / major_tick_freq) * major_tick_freq
            major_tick_max = np.floor(major_tick_max / major_tick_freq) * major_tick_freq
            n_major_ticks = int((major_tick_max - major_tick_min) / major_tick_freq) + 1
            major_ticks = np.linspace(major_tick_min, major_tick_max, n_major_ticks)
            major_tick_labels = [str(round(t, 3)) for t in major_ticks]     # Some weird str() conversion without rounding.
            axs[i].set_yticks(major_ticks)
            axs[i].set_yticklabels(major_tick_labels)

        # Set y axis minor ticks.
        if minor_tick_freq is not None:
            minor_tick_min = y_lim[0]
            if minor_tick_min is None:
                minor_tick_min = axs[i].get_ylim()[0]
            minor_tick_max = y_lim[1]
            if minor_tick_max is None:
                minor_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'minor_tick_freq'.
            minor_tick_min = np.ceil(minor_tick_min / minor_tick_freq) * minor_tick_freq
            minor_tick_max = np.floor(minor_tick_max / minor_tick_freq) * minor_tick_freq
            n_minor_ticks = int((minor_tick_max - minor_tick_min) / minor_tick_freq) + 1
            minor_ticks = np.linspace(minor_tick_min, minor_tick_max, n_minor_ticks)
            axs[i].set_yticks(minor_ticks, minor=True)

        # Set y grid lines.
        axs[i].grid(axis='y', linestyle='dashed', linewidth=linewidth)
        axs[i].set_axisbelow(True)

        # Set axis spine/tick linewidths and tick lengths.
        spines = ['top', 'bottom','left','right']
        for spine in spines:
            axs[i].spines[spine].set_linewidth(linewidth)
        axs[i].tick_params(which='both', length=ticklength, width=linewidth)

    # Set title.
    title_kwargs = {
        'fontsize': fontsize_title,
        'style': 'italic'
    }
    if title_x is not None:
        title_kwargs['x'] = title_x
    if title_y is not None:
        title_kwargs['y'] = title_y
    plt.title(title, **title_kwargs)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', dpi=dpi, pad_inches=0.03)
        logging.info(f"Saved plot to '{savepath}'.")

def plot_dataframe(
    ax: Optional[Axes] = None,
    data: Optional[pd.DataFrame] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    dpi: float = 1000,
    exclude_x: Optional[Union[str, List[str]]] = None,
    figsize: Tuple[float, float] = (16, 6),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontsize_label: Optional[float] = None,
    fontsize_legend: Optional[float] = None,
    fontsize_stats: Optional[float] = None,
    fontsize_tick_label: Optional[float] = None,
    fontsize_title: Optional[float] = None,
    hue_connections_index: Optional[Union[str, List[str]]] = None,
    hue_hatch: Optional[Union[str, List[str]]] = None,
    hue_label: Optional[Union[str, List[str]]] = None,
    hue_order: Optional[List[str]] = None,
    include_x: Optional[Union[str, List[str]]] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: str = 'upper right',
    linecolour: str = 'black',
    linewidth: float = 0.5,
    major_tick_freq: Optional[float] = None,
    minor_tick_freq: Optional[float] = None,
    n_cols: Optional[int] = None,
    outlier_legend_loc: str = 'upper left',
    palette: Optional[sns.palettes._ColorPalette] = sns.color_palette('colorblind'),
    pointsize: float = 10,
    savepath: Optional[str] = None,
    share_y: bool = False,
    show_boxes: bool = True,
    show_hue_connections: bool = False,
    show_hue_connections_inliers: bool = False,
    show_legend: bool = True,
    show_points: bool = True,
    show_stats: bool = False,
    show_x_tick_labels: bool = True,
    show_x_tick_label_counts: bool = True,
    stats_exclude: List[str] = [],
    stats_exclude_left: List[str] = [],
    stats_exclude_right: List[str] = [],
    stats_index: Optional[Union[str, List[str]]] = None,
    stats_paired: bool = True,
    stats_text_offset: Optional[float] = None,
    stats_two_sided: bool = False,
    style: Optional[Literal['box', 'violin']] = 'box',
    ticklength: float = 0.5,
    title: Optional[str] = None,
    title_x: Optional[float] = None,
    title_y: Optional[float] = None,
    x_label: Optional[str] = None,
    x_lim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None),
    x_order: Optional[List[str]] = None,
    x_width: float = 0.8,
    x_tick_label: Optional[List[str]] = None,
    x_tick_label_rot: float = 0,
    y_label: Optional[str] = None,
    y_lim: Optional[Tuple[Optional[float], Optional[float]]] = (None, None)):
    hue_hatches = arg_to_list(hue_hatch, str)
    hue_labels = arg_to_list(hue_label, str)
    include_xs = arg_to_list(include_x, str)
    exclude_xs = arg_to_list(exclude_x, str)
    if show_hue_connections and hue_connections_index is None:
        raise ValueError(f"Please set 'hue_connections_index' to allow matching points between hues.")
    if show_stats and stats_index is None:
        raise ValueError(f"Please set 'stats_index' to determine sample pairing for Wilcoxon test.")
    x_tick_labels = arg_to_list(x_tick_label, str)

    # Set default fontsizes.
    if fontsize_label is None:
        fontsize_label = fontsize
    if fontsize_legend is None:
        fontsize_legend = fontsize
    if fontsize_stats is None:
        fontsize_stats = fontsize
    if fontsize_tick_label is None:
        fontsize_tick_label = fontsize
    if fontsize_title is None:
        fontsize_title = fontsize
        
    # Include/exclude.
    if include_xs is not None:
        data = data[data[x].isin(include_xs)]
    if exclude_xs is not None:
        data = data[~data[x].isin(exclude_xs)]

    # Add outlier data.
    data = __add_outlier_info(data, x, y, hue)

    # Get min/max values for y-lim.
    if share_y:
        min_y = data[y].min()
        max_y = data[y].max()

    # Get x values.
    if x_order is None:
        x_order = list(sorted(data[x].unique()))

    # Determine x labels.
    groupby = x if hue is None else [x, hue]
    count_map = data.groupby(groupby)[y].count()
    if x_tick_labels is None:
        x_tick_labels = []
        for x_val in x_order:
            count = count_map.loc[x_val]
            if hue is not None:
                ns = count.values
                # Use a single number, e.g. (n=99) if all hues have the same number of points.
                if len(np.unique(ns)) == 1:
                    ns = ns[:1]
            else:
                ns = [count]
            label = f"{x_val}\n(n={','.join([str(n) for n in ns])})" if show_x_tick_label_counts else x_val
            x_tick_labels.append(label)

    # Create subplots if required.
    if n_cols is None:
        n_cols = len(x_order)
    n_rows = int(np.ceil(len(x_order) / n_cols))
    if ax is not None:
        assert n_rows == 1
        axs = [ax]
        # Figsize will have been handled externally.
    else:
        if n_rows > 1:
            _, axs = plt.subplots(n_rows, 1, dpi=dpi, figsize=(figsize[0], n_rows * figsize[1]), sharey=share_y)
        else:
            plt.figure(dpi=dpi, figsize=figsize)
            axs = [plt.gca()]

    # Get x-axis limits.
    x_lim = list(x_lim)
    if x_lim[0] is None:
        x_lim[0] = -0.5
    if x_lim[1] is None:
        x_lim[1] = n_cols - 0.5

    # Get hue order/colour/labels.
    if hue is not None:
        if hue_order is None:
            hue_order = list(sorted(data[hue].unique()))

        # Calculate x width for each hue.
        hue_width = x_width / len(hue_order) 

        # Check there are enough colours in palette.
        if len(hue_order) > len(palette):
            raise ValueError(f"'palette' doesn't have enough colours for hues '{hue_order}', needs '{len(hue_order)}'.")

        # Create map from hue to colour.
        hue_colours = dict((h, palette[i]) for i, h in enumerate(hue_order))

        if hue_labels is not None:
            if len(hue_labels) != len(hue_order):
                raise ValueError(f"Length of 'hue_labels' ({hue_labels}) should match hues ({hue_order}).")

    # Plot rows.
    for i in range(n_rows):
        # Split data.
        row_x_order = x_order[i * n_cols:(i + 1) * n_cols]
        row_x_tick_labels = x_tick_labels[i * n_cols:(i + 1) * n_cols]

        # Get x colours.
        if hue is None:
            # Check there are enough colors in palette.
            if len(row_x_order) > len(palette):
                raise ValueError(f"'palette' doesn't have enough colours for x values '{row_x_order}', needs '{len(row_x_order)}'.")
            x_colours = dict((x, palette[i]) for i, x in enumerate(row_x_order))

        # Get row data.
        row_data = data[data[x].isin(row_x_order)].copy()

        # Keep track of legend items.
        hue_artists = {}

        for j, x_val in enumerate(row_x_order):
            # Add x positions.
            if hue is not None:
                for k, hue_name in enumerate(hue_order):
                    x_pos = j - 0.5 * x_width + (k + 0.5) * hue_width
                    row_data.loc[(row_data[x] == x_val) & (row_data[hue] == hue_name), 'x_pos'] = x_pos
            else:
                x_pos = j
                row_data.loc[row_data[x] == x_val, 'x_pos'] = x_pos
                
            # Plot boxes.
            if show_boxes:
                if hue is not None:
                    for k, hue_name in enumerate(hue_order):
                        # Get hue data and pos.
                        hue_data = row_data[(row_data[x] == x_val) & (row_data[hue] == hue_name)]
                        if len(hue_data) == 0:
                            continue
                        hue_pos = hue_data.iloc[0]['x_pos']

                        # Get hue 'label' - allows us to use names more display-friendly than the data values.
                        hue_label = hue_name if hue_labels is None else hue_labels[k]

                        hatch = hue_hatches[k] if hue_hatches is not None else None
                        if style == 'box':
                            # Plot box.
                            res = axs[i].boxplot(hue_data[y].dropna(), boxprops=dict(color=linecolour, facecolor=hue_colours[hue_name], linewidth=linewidth), capprops=dict(color=linecolour, linewidth=linewidth), flierprops=dict(color=linecolour, linewidth=linewidth, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=linewidth), patch_artist=True, positions=[hue_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=linewidth), widths=hue_width)
                            if hatch is not None:
                                mpl.rcParams['hatch.linewidth'] = linewidth
                                res['boxes'][0].set_hatch(hatch)
                                # res['boxes'][0].set(hatch=hatch)
                                # res['boxes'][0].set_edgecolor('white')
                                # res['boxes'][0].set(facecolor='white')

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                        elif style == 'violin':
                            # Plot violin.
                            res = axs[i].violinplot(hue_data[y], positions=[hue_pos], widths=hue_width)

                            # Save reference to plot for legend.
                            if not hue_label in hue_artists:
                                hue_artists[hue_label] = res['boxes'][0]
                else:
                    # Plot box.
                    x_df = row_data[row_data[x] == x_val]
                    if len(x_df) == 0:
                        continue
                    x_pos = x_df.iloc[0]['x_pos']
                    if style == 'box':
                        axs[i].boxplot(x_df[y], boxprops=dict(color=linecolour, facecolor=x_colours[x_val], linewidth=linewidth), capprops=dict(color=linecolour, linewidth=linewidth), flierprops=dict(color=linecolour, linewidth=linewidth, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=linewidth), patch_artist=True, positions=[x_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=linewidth))
                    elif style == 'violin':
                        axs[i].violinplot(x_df[y], positions=[x_pos])

            # Plot points.
            if show_points:
                if hue is not None:
                    for j, hue_name in enumerate(hue_order):
                        hue_data = row_data[(row_data[x] == x_val) & (row_data[hue] == hue_name)]
                        if len(hue_data) == 0:
                            continue
                        res = axs[i].scatter(hue_data['x_pos'], hue_data[y], color=hue_colours[hue_name], edgecolors=linecolour, linewidth=linewidth, s=pointsize, zorder=100)
                        if not hue_label in hue_artists:
                            hue_artists[hue_label] = res
                else:
                    x_df = row_data[row_data[x] == x_val]
                    axs[i].scatter(x_df['x_pos'], x_df[y], color=x_colours[x_val], edgecolors=linecolour, linewidth=linewidth, s=pointsize, zorder=100)

            # Identify connections between hues.
            if hue is not None and show_hue_connections:
                # Get column/value pairs to group across hue levels.
                # line_ids = row_data[(row_data[x] == x_val) & row_data['outlier']][outlier_cols]
                x_df = row_data[(row_data[x] == x_val)]
                if not show_hue_connections_inliers:
                    line_ids = x_df[x_df['outlier']][hue_connections_index]
                else:
                    line_ids = x_df[hue_connections_index]

                # Drop duplicates.
                line_ids = line_ids.drop_duplicates()

                # Get palette.
                line_palette = sns.color_palette('husl', n_colors=len(line_ids))

                # Plot lines.
                artists = []
                labels = []
                for j, (_, line_id) in enumerate(line_ids.iterrows()):
                    # Get line data.
                    line_data = row_data[(row_data[x] == x_val)]
                    for k, v in zip(line_ids.columns, line_id):
                        line_data = line_data[line_data[k] == v]
                    line_data = line_data.sort_values('x_pos')
                    x_pos = line_data['x_pos'].tolist()
                    y_data = line_data[y].tolist()

                    # Plot line.
                    lines = axs[i].plot(x_pos, y_data, color=line_palette[j])

                    # Save line/label for legend.
                    artists.append(lines[0])
                    label = ':'.join(line_id.tolist())
                    labels.append(label)

                # Annotate outlier legend.
                if show_legend:
                    # Save main legend.
                    main_legend = axs[i].get_legend()

                    # Show outlier legend.
                    axs[i].legend(artists, labels, bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize_legend, loc=outlier_legend_loc)

                    # Re-add main legend.
                    axs[i].add_artist(main_legend)

        if hue is not None:
            if show_legend:
                # Filter 'hue_labels' based on hue 'artists'. Some hues may not be present in this row,
                # and 'hue_labels' is a global (across all rows) tracker.
                hue_labels = hue_order if hue_labels is None else hue_labels
                labels, artists = list(zip(*[(h, hue_artists[h]) for h in hue_labels if h in hue_artists]))

                # Show legend.
                legend = axs[i].legend(artists, labels, bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize_legend, loc=legend_loc)
                frame = legend.get_frame()
                frame.set_boxstyle('square')
                frame.set_edgecolor('black')
                frame.set_linewidth(linewidth)

        # Plot statistical significance.
        if hue is not None and show_stats:
            # Create pairs for stats annotation.
            pairs = []
            for x_val in row_x_order:
                pair = []
                for j, hue_val in enumerate(hue_order):
                    for hue_val_two in hue_order[j + 1:]:
                        pair = ((x_val, hue_val), (x_val, hue_val_two))
                        pairs.append(pair)

            # Calculate p-values.
            pairs_tmp = pairs.copy()
            pairs = []  # Some pairs may be removed if there is not data.
            p_vals = []
            for (x_a, hue_a), (x_b, hue_b) in pairs_tmp:
                assert x_a == x_b
                x_val = x_a
                x_df = row_data[row_data[x] == x_val]
                if x_val == 'Mandible':
                    x_df.to_csv('x.csv')
                x_df = x_df.pivot(index=stats_index, columns=[hue], values=[y]).reset_index()
                if x_val == 'Mandible':
                    x_df.to_csv('xp.csv')
                if (y, hue_a) in x_df.columns and (y, hue_b) in x_df.columns:
                    vals_a = x_df[y][hue_a]
                    vals_b = x_df[y][hue_b]
                    if not stats_paired:
                        vals_a = vals_a[~vals_a.isna()]
                        vals_b = vals_b[~vals_b.isna()]
                    pair = ((x_a, hue_a), (x_b, hue_b))
                    if stats_two_sided:
                        # Perform two-sided 'Wilcoxon signed rank test'.
                        if stats_paired:
                            _, p_val = wilcoxon(vals_a, vals_b, alternative='two-sided')
                        else:
                            _, p_val = mannwhitneyu(vals_a, vals_b, alternative='two-sided')
                        if p_val <= 0.05:
                            p_vals.append((p_val, ''))
                            pairs.append(pair)
                    else:
                        # Perform one-sided 'Wilcoxon signed rank test'.
                        if stats_paired:
                            _, p_val = wilcoxon(vals_a, vals_b, alternative='greater')
                        else:
                            _, p_val = mannwhitneyu(vals_a, vals_b, alternative='greater')
                        if p_val <= 0.05:
                            p_vals.append((p_val, '>'))
                            pairs.append(pair)
                        else:
                            if stats_paired:
                                _, p_val = wilcoxon(vals_a, vals_b, alternative='less')
                            else:
                                _, p_val = mannwhitneyu(vals_a, vals_b, alternative='less')
                            if p_val <= 0.05:
                                p_vals.append((p_val, '<'))
                                pairs.append(pair)

            # Exclude pairs.
            pairs_tmp = pairs.copy()
            p_vals_tmp = p_vals.copy()
            pairs = []
            p_vals = []
            for pair, p_val in zip(pairs_tmp, p_vals_tmp):
                (x_a, hue_a), (x_b, hue_b) = pair    
                if (hue_a in stats_exclude or hue_b in stats_exclude) or (hue_a in stats_exclude_left) or (hue_b in stats_exclude_right):
                    continue
                pairs.append(pair)
                p_vals.append(p_val)

            # Format p-values.
            p_vals = __format_p_values(p_vals) 

            # Annotate figure.
            if len(pairs) > 0:
                annotator = Annotator(axs[i], pairs, data=row_data, x=x, y=y, order=row_x_order, hue=hue, hue_order=hue_order, verbose=False)
                conf_kwargs = { 'fontsize': fontsize_stats, 'line_width': linewidth }
                if stats_text_offset is not None:
                    conf_kwargs['text_offset'] = stats_text_offset
                annotator.configure(**conf_kwargs)
                annotator.set_custom_annotations(p_vals)
                annotator.annotate()
          
        # Set axis labels.
        x_label = x_label if x_label is not None else ''
        y_label = y_label if y_label is not None else ''
        axs[i].set_xlabel(x_label, fontsize=fontsize_label)
        axs[i].set_ylabel(y_label, fontsize=fontsize_label)
                
        # Set axis tick labels.
        axs[i].set_xticks(list(range(len(row_x_tick_labels))))
        if show_x_tick_labels:
            axs[i].set_xticklabels(row_x_tick_labels, fontsize=fontsize_tick_label, rotation=x_tick_label_rot)
        else:
            axs[i].set_xticklabels([])

        axs[i].tick_params(axis='y', which='major', labelsize=fontsize_tick_label)

        # Set axis limits.
        axs[i].set_xlim(*x_lim)
        y_margin = 0.05
        if not share_y:
            min_y = row_data[y].min()
            max_y = row_data[y].max()
        y_lim_row = list(y_lim)
        if y_lim_row[0] is None:
            if y_lim_row[1] is None:
                width = max_y - min_y
                y_lim_row[0] = min_y - y_margin * width
                y_lim_row[1] = max_y + y_margin * width
            else:
                width = y_lim_row[1] - min_y
                y_lim_row[0] = min_y - y_margin * width
        else:
            if y_lim_row[1] is None:
                width = max_y - y_lim_row[0]
                y_lim_row[1] = max_y + y_margin * width
        axs[i].set_ylim(*y_lim_row)

        # Set y axis major ticks.
        if major_tick_freq is not None:
            major_tick_min = y_lim[0]
            if major_tick_min is None:
                major_tick_min = axs[i].get_ylim()[0]
            major_tick_max = y_lim[1]
            if major_tick_max is None:
                major_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'major_tick_freq'.
            major_tick_min = np.ceil(major_tick_min / major_tick_freq) * major_tick_freq
            major_tick_max = np.floor(major_tick_max / major_tick_freq) * major_tick_freq
            n_major_ticks = int((major_tick_max - major_tick_min) / major_tick_freq) + 1
            major_ticks = np.linspace(major_tick_min, major_tick_max, n_major_ticks)
            major_tick_labels = [str(round(t, 3)) for t in major_ticks]     # Some weird str() conversion without rounding.
            axs[i].set_yticks(major_ticks)
            axs[i].set_yticklabels(major_tick_labels)

        # Set y axis minor ticks.
        if minor_tick_freq is not None:
            minor_tick_min = y_lim[0]
            if minor_tick_min is None:
                minor_tick_min = axs[i].get_ylim()[0]
            minor_tick_max = y_lim[1]
            if minor_tick_max is None:
                minor_tick_max = axs[i].get_ylim()[1]
            
            # Round range to nearest multiple of 'minor_tick_freq'.
            minor_tick_min = np.ceil(minor_tick_min / minor_tick_freq) * minor_tick_freq
            minor_tick_max = np.floor(minor_tick_max / minor_tick_freq) * minor_tick_freq
            n_minor_ticks = int((minor_tick_max - minor_tick_min) / minor_tick_freq) + 1
            minor_ticks = np.linspace(minor_tick_min, minor_tick_max, n_minor_ticks)
            axs[i].set_yticks(minor_ticks, minor=True)

        # Set y grid lines.
        axs[i].grid(axis='y', linestyle='dashed', linewidth=linewidth)
        axs[i].set_axisbelow(True)

        # Set axis spine/tick linewidths and tick lengths.
        spines = ['top', 'bottom','left','right']
        for spine in spines:
            axs[i].spines[spine].set_linewidth(linewidth)
        axs[i].tick_params(which='both', length=ticklength, width=linewidth)

    # Set title.
    title_kwargs = {
        'fontsize': fontsize_title,
        'style': 'italic'
    }
    if title_x is not None:
        title_kwargs['x'] = title_x
    if title_y is not None:
        title_kwargs['y'] = title_y
    plt.title(title, **title_kwargs)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', dpi=dpi, pad_inches=0.03)
        logging.info(f"Saved plot to '{savepath}'.")

def plot_registration(
    fixed_id: str,
    moving_id: str,
    fixed_size: ImageSize3D,
    moving_size: ImageSize3D,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    axs: Optional[Axes] = None,
    cca: bool = False,
    colour: Optional[Union[str, List[str]]] = None,
    dose_alpha: float = 0.3,
    dose_data: Optional[np.ndarray] = None,
    dose_legend_size: float = 0.03,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'region_data' if 'str', else 'np.ndarray') and min/max of extent.
    figsize: Tuple[int, int] = (12, 9),
    fixed_centre_of: Optional[Union[str, np.ndarray]] = None,             # Uses 'fixed_region_data' if 'str', else uses 'np.ndarray'.
    fixed_crop: Optional[Union[str, np.ndarray, Crop2D]] = None,    # Uses 'fixed_region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    fixed_crop_margin: float = 100,                                       # Applied if cropping to 'fixed_region_data' or 'np.ndarray'
    fixed_ct_data: Optional[np.ndarray] = None,
    fixed_region_data: Optional[np.ndarray] = None,
    fixed_slice_idx: Optional[int] = None,
    fixed_spacing: ImageSpacing3D = None,
    fontsize: int = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_show_all_regions: bool = False,
    linestyle_region: bool = 'solid',
    moving_centre_of: Optional[Union[str, np.ndarray]] = None,             # Uses 'moving_region_data' if 'str', else uses 'np.ndarray'.
    moving_crop: Optional[Union[str, np.ndarray, Crop2D]] = None,    # Uses 'moving_region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moving_crop_margin: float = 100,                                       # Applied if cropping to 'moving_region_data' or 'np.ndarray'
    moving_ct_data: Optional[np.ndarray] = None,
    moving_region_data: Optional[np.ndarray] = None,
    moving_slice_idx: Optional[int] = None,
    moving_spacing: ImageSpacing3D = None,
    norm: Optional[Tuple[float, float]] = None,
    perimeter: bool = True,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    registered_centre_of: Optional[Union[str, np.ndarray]] = None,             # Uses 'registered_region_data' if 'str', else uses 'np.ndarray'.
    registered_crop: Optional[Union[str, np.ndarray, Crop2D]] = None,    # Uses 'registered_region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    registered_crop_margin: float = 100,                                       # Applied if cropping to 'registered_region_data' or 'np.ndarray'
    registered_ct_data: Optional[np.ndarray] = None,
    registered_region_data: Optional[np.ndarray] = None,
    registered_slice_idx: Optional[int] = None,
    registered_use_fixed_crop: bool = True,
    registered_use_fixed_slice: bool = True,
    savepath: Optional[str] = None,
    show: bool = True,
    show_axis: bool = True,
    show_extent: bool = False,
    show_legend: bool = True,
    show_title: bool = True,
    show_x_label: bool = True,
    show_x_ticks: bool = True,
    show_y_label: bool = True,
    show_y_ticks: bool = True,
    title: Optional[str] = None,
    transform: torchio.transforms.Transform = None,
    view: Axis = 0,
    window: Optional[Union[Literal['bone', 'lung', 'tissue'], Tuple[float, float]]] = 'tissue') -> None:
    __assert_slice_idx(fixed_centre_of, extent_of, fixed_slice_idx)
    __assert_slice_idx(moving_centre_of, extent_of, moving_slice_idx)
    assert view in (0, 1, 2)
    centres_of = (fixed_centre_of, registered_centre_of, moving_centre_of)
    crops = (fixed_crop, registered_crop, moving_crop)
    crop_margins = (fixed_crop_margin, registered_crop_margin, moving_crop_margin)
    ct_data = (fixed_ct_data, registered_ct_data, moving_ct_data)
    ct_sizes = (fixed_size, fixed_size, moving_size)
    ids = (fixed_id, f'{moving_id} -> {fixed_id}', moving_id)
    region_data = (fixed_region_data, registered_region_data, moving_region_data)
    slice_idxs = (fixed_slice_idx, registered_slice_idx, moving_slice_idx)
    spacings = (fixed_spacing, fixed_spacing, moving_spacing)
    infos = ({}, { 'fixed': registered_use_fixed_slice }, {})

    if axs is None:
        # Create figure/axes.
        # fig, axs = plt.subplots(3, 1, figsize=figsize, layout='tight', gridspec_kw={'hspace': 0.1})
        # fig.subplots_adjust(hspace=0)
        _, axs = plt.subplots(2, 2, figsize=figsize, gridspec_kw={ 'hspace': 0.3 })
        axs[1][1].set_visible(False)
        axs = axs.flatten()
        close_figure = True
    else:
        # Assume that parent routine will call 'plt.show()' after all axes are plotted.
        assert len(axs) == 3
        close_figure = False
        show = False

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Get slice indexes.
    slice_idxs_tmp = []
    for i, (centre_of, slice_idx, data) in enumerate(zip(centres_of, slice_idxs, region_data)):
        if registered_use_fixed_slice and i == 1:
            slice_idxs_tmp.append(slice_idxs_tmp[0])
            continue

        if slice_idx is None:
            label = data[centre_of] if type(centre_of) == str else centre_of
            if label.sum() == 0:
                raise ValueError(f"'centre_of' array must not be empty.")
            extent_centre = get_extent_centre(label)
            slice_idx = extent_centre[view]
            slice_idxs_tmp.append(slice_idx)
        else:
            slice_idxs_tmp.append(slice_idx)
    slice_idxs = slice_idxs_tmp

    # Convert each 'crop' to 'Box2D' type.
    crops_tmp = []
    for i, (crop, data, margin, spacing) in enumerate(zip(crops, region_data, crop_margins, spacings)):
        if registered_use_fixed_crop and i == 1:
            crops_tmp.append(crops_tmp[0])
            continue

        if crop is not None:
            if type(crop) == str:
                crop = __get_region_crop(data[crop], margin, spacing, view)     # Crop was 'region_data' key.
            elif type(crop) == np.ndarray:
                crop = __get_region_crop(crop, margin, spacing, view)                  # Crop was 'np.ndarray'.
            else:
                crop = tuple(zip(*crop))                                                    # Crop was 'Crop2D' type.
        crops_tmp.append(crop)
    crops = crops_tmp

    # Get CT slices.
    ct_slice_data = []
    for i, (ct, slice_idx, ct_size) in enumerate(zip(ct_data, slice_idxs, ct_sizes)):
        if ct is not None:
            ct_slice_data.append(__get_slice_data(ct, slice_idx, view))
        else:
            ct_slice_data.append(__get_slice_data(np.zeros(shape=ct_size), slice_idx, view))

    # Crop CT slice data.
    ct_slice_data_tmp = []
    for ct_slice, crop in zip(ct_slice_data, crops):
        if crop is not None:
            ct_slice_data_tmp.append(crop_2D(ct_slice, __reverse_box_coords_2D(crop)))
        else:
            ct_slice_data_tmp.append(ct_slice)
    ct_slice_data = ct_slice_data_tmp

    # Determine aspect ratio.
    aspects = []
    for i, spacing in enumerate(spacings):
        aspect = __get_aspect_ratio(view, spacing)
        aspects.append(aspect)

    # Determine CT window.
    vmins = []
    vmaxs = []
    for i, ct in enumerate(ct_data):
        if ct_data is None:
            vmins.append(None)
            vmaxs.append(None)
            continue
        
        if window is not None:
            if type(window) == str:
                if window == 'bone':
                    width, level = (2000, 300)
                elif window == 'lung':
                    width, level = (2000, -200)
                elif window == 'tissue':
                    width, level = (350, 50)
                else:
                    raise ValueError(f"Window '{window}' not recognised.")
            else:
                width, level = window
            vmin = level - (width / 2)
            vmax = level + (width / 2)
        else:
            vmin = ct_data.min()
            vmax = ct_data.max()

        vmins.append(vmin)
        vmaxs.append(vmax)

    # Plot CT data.
    for ax, slice, aspect, vmin, vmax in zip(axs, ct_slice_data, aspects, vmins, vmaxs):
        ax.imshow(slice, cmap='gray', aspect=aspect, interpolation='none', origin=__get_origin(view), vmin=vmin, vmax=vmax)

    # Hide axes.
    if not show_axis:
        for ax in axs:
            ax.set_axis_off()

    # Add 'x-axis' label.
    if show_x_label:
        for ax, spacing in zip(axs, spacings):
            if view == 0:
                spacing_x = spacing[1]
            elif view == 1: 
                spacing_x = spacing[0]
            elif view == 2:
                spacing_x = spacing[0]

            ax.set_xlabel(f'voxel [@ {spacing_x:.3f} mm spacing]')

    # Add 'y-axis' label.
    if show_y_label:
        for ax, spacing in zip(axs, spacings):
            if view == 0:
                spacing_y = spacing[2]
            elif view == 1:
                spacing_y = spacing[2]
            elif view == 2:
                spacing_y = spacing[1]

            ax.set_ylabel(f'voxel [@ {spacing_y:.3f} mm spacing]')

    # Plot region data.
    for ax, data, slice_idx, crop, aspect in zip(axs, region_data, slice_idxs, crops, aspects):
        if data is not None:
            # Plot regions.
            should_show_legend = __plot_region_data(data, slice_idx, alpha_region, aspect, latex, perimeter, view, ax=ax, cca=cca, colour=colour, crop=crop, legend_show_all_regions=legend_show_all_regions, linestyle=linestyle_region, show_extent=show_extent)

            # Create legend.
            if show_legend and should_show_legend:
                plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
                for l in plt_legend.get_lines():
                    l.set_linewidth(8)

    # Set axis limits if cropped.
    for ax, crop, slice in zip(axs, crops, ct_slice_data):
        if crop is not None:
            # Get new x ticks/labels.
            x_diff = int(np.diff(ax.get_xticks())[0])
            x_tick_label_max = int(np.floor(crop[1][0] / x_diff) * x_diff)
            x_tick_labels = list(range(crop[0][0], x_tick_label_max, x_diff))
            x_ticks = list((i * x_diff for i in range(len(x_tick_labels))))

            # Round up to nearest 'x_diff'.
            x_tick_labels_tmp = x_tick_labels.copy()
            x_tick_labels = [int(np.ceil(x / x_diff) * x_diff) for x in x_tick_labels_tmp]
            x_tick_diff = list(np.array(x_tick_labels) - np.array(x_tick_labels_tmp))
            x_ticks = list(np.array(x_ticks) + np.array(x_tick_diff))

            # Set x ticks.
            ax.set_xticks(x_ticks, labels=x_tick_labels)

            # Get new y ticks/labels.
            y_diff = int(np.diff(ax.get_yticks())[0])
            y_tick_label_max = int(np.floor(crop[1][1] / y_diff) * y_diff)
            y_tick_labels = list(range(crop[0][1], y_tick_label_max, y_diff))
            y_ticks = list((i * y_diff for i in range(len(y_tick_labels))))

            # Round up to nearest 'y_diff'.
            y_tick_labels_tmp = y_tick_labels.copy()
            y_tick_labels = [int(np.ceil(y / y_diff) * y_diff) for y in y_tick_labels_tmp]
            y_tick_diff = list(np.array(y_tick_labels) - np.array(y_tick_labels_tmp))
            y_ticks = list(np.array(y_ticks) + np.array(y_tick_diff))

            # Set y ticks.
            ax.set_yticks(y_ticks, labels=y_tick_labels)

    # Show axis ticks.
    if not show_x_ticks:
        for ax in axs:
            ax.get_xaxis().set_ticks([])

    if not show_y_ticks:
        for ax in axs:
            ax.get_yaxis().set_ticks([])

    # Add title.
    if show_title:
        for ax, id, slice_idx, ct_size, info in zip(axs, ids, slice_idxs, ct_sizes, infos):
            n_slices = ct_size[view]
            slice_info = ' (fixed)' if 'fixed' in info and info['fixed'] else ''
            title = f"patient: {id}, slice{slice_info}: {slice_idx}/{n_slices - 1} ({__view_to_text(view)} view)"

            # Escape text if using latex.
            if latex:
                title = __escape_latex(title)

            ax.set_title(title)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show:
        plt.show()

        # Revert latex settings.
        if latex:
            plt.rcParams.update({
                "font.family": rc_params['font.family'],
                'text.usetex': rc_params['text.usetex']
            })

    if close_figure:
        plt.close() 

def style_rows(
    series: pd.Series,
    col_groups: Optional[List[str]] = None,
    exclude_cols: Optional[List[str]] = None) -> List[str]:
    styles = []
    if col_groups is not None:
        for col_group in col_groups:
            styles += __get_styles(series[col_group], exclude_cols=exclude_cols)
    else:
        styles += __get_styles(series, exclude_cols=exclude_cols)
    return styles

def __add_outlier_info(df, x, y, hue):
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

def __format_p_values(
    p_vals: List[float],
    show_direction: bool = True) -> List[str]:
    # Format p value for display.
    p_vals_tmp = p_vals
    p_vals = []
    for p_val, direction in p_vals_tmp:
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

        # Add direction if necessary.
        if show_direction and direction != '':
            p_val = f'{p_val} {direction}'
        p_vals.append(p_val)

    return p_vals

def __get_styles(
    series: pd.Series,
    exclude_cols: Optional[List[str]] = None) -> List[str]:
    null_colour = 'background-color: #FFFFE0'

    # Normalise values.
    vals = []
    for index, value in series.iteritems():
        if np.isnan(value) or index in exclude_cols:
            continue
        else:
            vals.append(value)
    val_range = (np.max(vals) - np.min(vals))
    if val_range == 0:
        return [null_colour] * len(series)
    slope = 1 / (val_range)
    offset = -np.min(vals)

    # Add styles based upon values.
    styles = []
    cmap = plt.cm.get_cmap('PuBu')
    for index, value in series.iteritems():
        if np.isnan(value) or index in exclude_cols:
            styles.append(null_colour)
        else:
            # Apply gradient colour.
            value = slope * (value + offset)
            colour = cmap(value)
            colour = rgb2hex(colour)
            styles.append(f'background-color: {colour}')

    return styles

def __get_view_size(
    view: Axis,
    size: ImageSize3D) -> ImageSize2D:
    if view == 0:
        return (size[1], size[2])
    elif view == 1:
        return (size[0], size[2])
    elif view == 2:
        return (size[0], size[1])

def __get_view_spacing(
    view: Axis,
    spacing: ImageSpacing3D) -> ImageSpacing2D:
    if view == 0:
        return (spacing[1], spacing[2])
    elif view == 1:
        return (spacing[0], spacing[2])
    elif view == 2:
        return (spacing[0], spacing[1])

def __view_to_text(view: int) -> str:
    if view == 0:
        return 'sagittal'
    elif view == 1:
        return 'coronal'
    elif view == 2:
        return 'axial'

from collections.abc import Iterable
from enum import Enum
import itk
import matplotlib as mpl
from matplotlib.colors import ListedColormap, rgb2hex, to_rgb
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
import scipy
import seaborn as sns
from seaborn.palettes import _ColorPalette
import SimpleITK as sitk
import torchio
from typing import *

from mymi.geometry import get_box, extent, centre_of_extent
from mymi import logging
from mymi.postprocessing import largest_cc_3D
from mymi.regions import get_region_patch_size
from mymi.regions import truncate_spine as truncate
from mymi.transforms import crop_or_pad_box, crop_point, crop, itk_transform_image, sitk_transform_image
from mymi.typing import *
from mymi.utils import arg_to_list

DEFAULT_FONT_SIZE = 10

class AltHyp(Enum):
    LESSER = 0
    GREATER = 1
    TWO_SIDED = 2

def plot_histogram(
    data: np.ndarray,
    n_bins: int = 100,
    title: Optional[str] = None,
    x_lim: Optional[Tuple[Optional[float], Optional[float]]] = None) -> None:
    plt.hist(data, bins=n_bins)

    if x_lim is not None:
        plt.xlim(x_lim)
    if title is not None:
        plt.title(title)

    # Add stats box.
    text = f"min/max: {data.min():.2f}/{data.max():.2f}\n\
        mean: {data.mean():.2f}\n\
        median: {np.median(data):.2f}\n\
        stdev: {data.std():.2f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.95, 0.95, text, bbox=props, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right')

    plt.show()

def __plot_region_data(
    data: RegionLabels,
    ax: mpl.axes.Axes,
    idx: int,
    aspect: float,
    alpha: float = 0.3,
    colours: Optional[Union[str, List[str]]] = None,
    crop: Optional[Box2D] = None,
    escape_latex: bool = False,
    legend_show_all_regions: bool = False,
    show_extent: bool = False,
    show_boundary: bool = True,
    use_cca: bool = False,
    view: Axis = 0) -> bool:
    __assert_view(view)

    regions = list(data.keys()) 
    if colours is None:
        colours = sns.color_palette('colorblind', n_colors=len(regions))
    else:
        colours = arg_to_list(colours, (str, tuple))

    if not ax:
        ax = plt.gca()

    # Plot each region.
    show_legend = False
    for region, colour in zip(regions, colours):
        # Define cmap.
        cmap = ListedColormap(((1, 1, 1, 0), colour))

        # Convert data to 'imshow' co-ordinate system.
        slice_data = __get_mpl_slice(data[region], idx, view)

        # Crop image.
        if crop:
            slice_data = crop(slice_data, __reverse_box_coords_2D(crop))

        # Plot extent.
        if show_extent:
            ext = extent(data[region])
            if ext is not None:
                label = f'{region} extent' if __box_in_plane(ext, view, idx) else f'{region} extent (offscreen)'
                __plot_box_slice(ext, view, ax=ax, colour=colour, crop=crop, label=label, linestyle='dashed')
                show_legend = True

        # Skip region if not present on this slice.
        if not legend_show_all_regions and slice_data.max() == 0:
            continue
        else:
            show_legend = True

        # Get largest component.
        if use_cca:
            slice_data = largest_cc_3D(slice_data)

        # Plot region.
        ax.imshow(slice_data, alpha=alpha, aspect=aspect, cmap=cmap, interpolation='none', origin=__get_mpl_origin(view))
        label = __escape_latex(region) if escape_latex else region
        ax.plot(0, 0, c=colour, label=label)
        ax.contour(slice_data, colors=[colour], levels=[.5], linestyles='solid')

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

def __get_mpl_origin(view: Axis) -> str:
    if view == 2:
        return 'upper'
    else:
        return 'lower'

def __get_mpl_slice(
    data: np.ndarray,
    idx: int,
    view: Axis) -> np.ndarray:
    # Check that slice index isn't too large.
    if idx >= data.shape[view]:
        raise ValueError(f"Idx '{idx}' out of bounds, only '{data.shape[view]}' {__view_to_text(view)} indices.")

    # Get correct plane.
    data_index = (
        idx if view == 0 else slice(data.shape[0]),
        idx if view == 1 else slice(data.shape[1]),
        idx if view == 2 else slice(data.shape[2]),
    )
    slice_data = data[data_index]

    # Convert from our coordinate system (LPS) to 'imshow' coords.
    slice_data = np.transpose(slice_data)

    return slice_data

def __get_mpl_aspect(
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
    idx: int) -> bool:
    # Get view bounding box.
    min, max = box
    min = min[view]
    max = max[view]

    # Calculate if the box is in plane.
    result = idx >= min and idx <= max
    return result

def __plot_box_slice(
    box: Box3D,
    view: Axis,
    ax: Optional[mpl.axes.Axes] = None,
    colour: str = 'r',
    crop: Box2D = None,
    label: str = 'box',
    linestyle: str = 'solid') -> None:
    if ax is None:
        ax = plt.gca()

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
    ax.add_patch(rect)
    ax.plot(0, 0, c=colour, label=label, linestyle=linestyle)

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

def __assert_dataframe(data: pd.DataFrame) -> None:
    if len(data) == 0:
        raise ValueError("Dataframe is empty.")

def __assert_data_type(
    name: str,
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    dtype: str) -> None:
    if isinstance(data, np.ndarray):
        if data.dtype != dtype:
            raise ValueError(f"Data {name} must be of type '{dtype}', got '{data.dtype}'.")
    if isinstance(data, dict):
        for key, value in data.items():
            if value.dtype != dtype:
                raise ValueError(f"Data {name} must be of type '{dtype}', got '{value.dtype}' for key '{key}'.")

def __assert_idx(
    centre: Optional[int],
    extent_of: Optional[Tuple[str, bool]],
    idx: Optional[float]) -> None:
    if centre is None and extent_of is None and idx is None:
        raise ValueError(f"Either 'centre', 'extent_of' or 'idx' must be set.")
    elif (centre is not None and extent_of is not None) or (centre is not None and idx is not None) or (extent_of is not None and idx is not None):
        raise ValueError(f"Only one of 'centre', 'extent_of' or 'idx' can be set.")

def __assert_view(view: int) -> None:
    assert view in (0, 1, 2)

def plot_heatmap(
    id: str,
    heatmap: np.ndarray,
    spacing: ImageSpacing3D,
    alpha_heatmap: float = 0.7,
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    crop_margin: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    fontsize: int = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    pred_data: Optional[Dict[str, np.ndarray]] = None,
    region_data: Optional[Dict[str, np.ndarray]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_colorbar: bool = True,
    show_legend: bool = True,
    idx: Optional[int] = None,
    view: Axis = 0, 
    **kwargs) -> None:
    __assert_idx(centre, extent_of, idx)

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

    if centre is not None:
        # Get 'idx' at centre of data.
        label = region_data[centre] if type(centre) == str else centre
        extent_centre = centre_of_extent(label)
        idx = extent_centre[view]

    if extent_of is not None:
        # Get 'idx' at min/max extent of data.
        label = region_data[extent_of[0]] if type(extent_of[0]) == str else extent_of
        extent_end = 0 if extent_of[1] == 'min' else 1
        ext = extent(label)
        idx = ext[extent_end][view]

    # Plot patient regions.
    size = heatmap.shape
    plot_patients_matrix(id, size, spacing, alpha_region=alpha_region, aspect=aspect, ax=ax, crop=crop, crop_margin=crop_margin, ct_data=ct_data, latex=latex, legend_loc=legend_loc, region_data=region_data, show=False, show_legend=False, idx=idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'Box2D' type.
        if type(crop) == str:
            crop = __get_region_crop(region_data[crop], crop_margin, spacing, view)     # Crop was 'region_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Box2D' type.

    # Get aspect ratio.
    if not aspect:
        aspect = __get_mpl_aspect(view, spacing) 

    # Get slice data.
    heatmap_slice = __get_mpl_slice(heatmap, idx, view)

    # Crop the image.
    if crop is not None:
        heatmap_slice = crop(heatmap_slice, __reverse_box_coords_2D(crop))

    # Plot heatmap
    image = ax.imshow(heatmap_slice, alpha=alpha_heatmap, aspect=aspect, origin=__get_mpl_origin(view))
    if show_colorbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.02)
        plt.colorbar(image, cax=cax)

    # Plot predictions.
    if pred_data is not None:
        for pred_label, pred in pred_data.items():
            if pred.sum() != 0:
                # Get slice data.
                pred_slice = __get_mpl_slice(pred, idx, view)

                # Crop the image.
                if crop:
                    pred_slice = crop(pred_slice, __reverse_box_coords_2D(crop))

                # Plot prediction.
                if pred_slice.sum() != 0: 
                    cmap = ListedColormap(((1, 1, 1, 0), colour))
                    ax.imshow(pred_slice, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_mpl_origin(view))
                    ax.plot(0, 0, c=colour, label=pred_label)
                    ax.contour(pred_slice, colors=[colour], levels=[.5], linestyles='solid')

            # Plot prediction extent.
            if pred.sum() != 0 and show_pred_extent:
                # Get prediction extent.
                pred_extent = extent(pred)

                # Plot if extent box is in view.
                label = f'{model_name} extent' if __box_in_plane(pred_extent, view, z) else f'{model_name} extent (offscreen)'
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

def plot_patients_matrix(
    # Allows us to plot multiple patients (rows) and patient views (columns).
    # Difficult to extent do patient studies on the columns - the data lists are
    # for different patients. E.g. list[ct_data] shows data for each patient,
    # and plot_single_patient_view can slice this data using the view it wants.
    # For multiple studies along the columns, we'd have to pass multiple ct data
    # per patient.
    plot_ids: Union[str, List[str]],
    spacings: Union[ImageSpacing3D, List[ImageSpacing3D]],
    ax: Optional[mpl.axes.Axes] = None,
    centres: Optional[Union[PatientLandmark, PatientRegion, Landmarks, RegionLabel, List[Union[PatientLandmark, PatientRegion, Landmarks, RegionLabel]]]] = None,
    crops: Optional[Union[str, np.ndarray, Box2D, List[Union[str, np.ndarray, Box2D]]]] = None,
    ct_datas: Optional[Union[CtImage, List[CtImage]]] = None,
    figsize: Tuple[int, int] = (10, 10),
    landmark_datas: Optional[Union[Landmarks, List[Landmarks]]] = None,
    region_datas: Optional[Union[RegionLabels, List[RegionLabels]]] = None,
    views: Union[Axis, List[Axis], Literal['all']] = 0,
    **kwargs) -> None:
    # Broadcast args to length of plot_ids.
    plot_ids = arg_to_list(plot_ids, str)
    n_rows = len(plot_ids)
    spacings = arg_to_list(spacings, ImageSpacing3D, length=n_rows)
    centres = arg_to_list(centres, (None, PatientLandmark, PatientRegion, Landmarks, RegionLabel), length=n_rows)
    crops = arg_to_list(crops, (None, str, np.ndarray, Box2D), length=n_rows)
    ct_datas = arg_to_list(ct_datas, (None, CtImage), length=n_rows)
    landmark_datas = arg_to_list(landmark_datas, (None, Landmarks), length=n_rows)
    region_datas = arg_to_list(region_datas, (None, RegionLabels), length=n_rows)
    views = arg_to_list(views, (int, Axis))
    if views == 'all':
        views = tuple(list(range(3)))
    n_cols = len(views)

    # Convert figsize from cm to inches.
    figsize = __convert_figsize_to_inches(figsize)

    # Create axes.
    if ax is None:
        if n_rows > 1 or n_cols > 1:
            # Subplots for multiple views.
            _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize[0], n_rows * figsize[1]), gridspec_kw={ 'hspace': 0.4 })
            if n_rows == 1:
                axs = [axs]
            elif n_cols == 1:
                axs = [[a] for a in axs]
        else:
            plt.figure(figsize=figsize)
            ax = plt.axes(frameon=False)
            axs = [[ax]]
    else:
        axs = [[ax]]

    # Only show figure if 'ax' wasn't passed.
    show_figure, close_figure = ax is None, ax is None

    # Plot each plot.
    show_figure_i = False
    close_figure_i = False
    for i in range(n_rows):
        for j in range(n_cols):
            if i == n_rows - 1 and j == n_cols - 1:
                show_figure_i = show_figure
                close_figure_i = close_figure

            plot_patient(
                plot_ids[i],
                ct_datas[i].shape,
                spacings[i],
                ax=axs[i][j],
                centre=centres[i],
                close_figure=close_figure_i,
                crop=crops[i],
                ct_data=ct_datas[i],
                landmark_data=landmark_datas[i],
                region_data=region_datas[i],
                show_figure=show_figure_i,
                view=views[j],
                **kwargs)

def plot_patient(
    plot_id: str,
    size: ImageSize3D,
    spacing: ImageSpacing3D,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[Union[PatientLandmark, PatientRegion, Landmarks, RegionLabel]] = None,
    close_figure: bool = True,
    colours: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[str, np.ndarray, Box2D]] = None,    # Uses 'region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    crop_margin: float = 100,                                       # Applied if cropping to 'region_data' or 'np.ndarray'.
    ct_data: Optional[np.ndarray] = None,
    dose_alpha_min: float = 0.1,
    dose_alpha_max: float = 0.7,
    dose_cmap: str = 'rainbow',
    dose_colourbar_pad: float = 0.05,
    dose_colourbar_size: float = 0.03,
    dose_data: Optional[np.ndarray] = None,
    escape_latex: bool = False,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'region_data' if 'str', else 'np.ndarray') and min/max of extent.
    fontsize: int = DEFAULT_FONT_SIZE,
    idx: Optional[float] = None,
    landmark_data: Optional[Landmarks] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_show_all_regions: bool = False,
    linewidth: float = 0.5,
    linewidth_legend: float = 8,
    norm: Optional[Tuple[float, float]] = None,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    region_data: Optional[RegionLabels] = None,            # All data passed to 'region_data' is plotted.
    savepath: Optional[str] = None,
    show_axes: bool = True,
    show_ct: bool = True,
    show_dose_bar: bool = True,
    show_extent: bool = False,
    show_figure: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    show_title_idx: bool = True,
    show_title_view: bool = True,
    show_x_label: bool = True,
    show_x_ticks: bool = True,
    show_y_label: bool = True,
    show_y_ticks: bool = True,
    title: Optional[str] = None,
    transform: torchio.transforms.Transform = None,
    view: Axis = 0,
    window: Optional[Union[Literal['bone', 'lung', 'tissue'], Tuple[float, float]]] = 'tissue',
    **kwargs) -> None:
    __assert_idx(centre, extent_of, idx)
    assert view in (0, 1, 2)

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if escape_latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    idx = __convert_float_idx(idx, size, view)

    # Get slice idx.
    if centre is not None:
        if isinstance(centre, str):
            if landmark_data is not None and centre in landmark_data['landmark-id']:
                centre_point = tuple(landmark_data[landmark_data['landmark-id'] == centre][list(range(3))].iloc[0])
            elif region_data is not None and centre in region_data:
                centre_point = __get_region_centre(region_data[centre])
            else:
                raise ValueError(f"No region/landmark '{centre}' found in data.")
        elif isinstance(centre, Landmarks):
            centre_point = tuple(landmark_data[list(range(3))].iloc[0])
        elif isinstance(centre, RegionLabel):
            centre_point = __get_region_centre(centre)
        idx = centre_point[view]

    # Get idx at min/max extent of label.
    if extent_of is not None:
        if len(extent_of) == 2:
            eo_region, eo_end = extent_of
            eo_axis = view
        elif len(extent_of) == 3:
            eo_region, eo_end, eo_axis = extent_of

        label = region_data[eo_region] if type(eo_region) == str else eo_region     # 'eo_region' can be str ('region_data' key) or np.ndarray.
        assert eo_end in ('min', 'max'), "'extent_of' must have one of ('min', 'max') as second element."
        eo_end = 0 if eo_end == 'min' else 1
        if postproc:
            label = postproc(label)
        ext_vox = extent(label, eo_axis, eo_end, view)
        idx = ext_vox[eo_end][axis]

    # Convert 'crop' to 'Box2D' type.
    if crop is not None:
        if isinstance(crop, str):
            if landmark_data is not None and crop in landmark_data['landmark-id']:
                lm_data = landmark_data[landmark_data['landmark-id'] == crop]
                crop = __get_landmark_crop_box(lm_data, crop_margin, size, spacing, view)
            elif region_data is not None and crop in region_data:
                crop = __get_region_crop_box(region_data[crop], crop_margin, spacing, view)
        elif isinstance(crop, Landmarks):
            crop = __get_landmark_crop_box(crop, crop_margin, size, spacing, view)
        elif isinstance(centre, RegionLabel):
            crop = __get_region_crop_box(crop, crop_margin, spacing, view)
        else:
            # Crop is passed as ((x_min, x_max), (y_min, y_max)) but box uses
            # ((x_min, y_min), (x_max, y_max)) format.
            crop = tuple(*zip(crop))

    if ct_data is not None:
        # Perform any normalisation.
        if norm is not None:
            mean, std_dev = norm
            
        # Load CT slice.
        ct_slice = __get_mpl_slice(ct_data, idx, view)
        if dose_data is not None:
            dose_slice = __get_mpl_slice(dose_data, idx, view)
    else:
        # Load empty slice.
        ct_slice = __get_mpl_slice(np.zeros(shape=size), idx, view)

    # Perform crop on CT data or placeholder.
    if crop is not None:
        ct_slice = crop(ct_slice, __reverse_box_coords_2D(crop))

        if dose_data is not None:
            dose_slice = crop(dose_slice, __reverse_box_coords_2D(crop))

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = __get_mpl_aspect(view, spacing) 

    # Determine CT window.
    if ct_data is not None:
        if window is not None:
            if type(window) == str:
                if window == 'bone':
                    width, level = (1800, 400)
                elif window == 'extent':
                    width = ct_data.max() - ct_data.min()
                    level = (ct_data.min() + ct_data.max()) / 2
                elif window == 'lung':
                    width, level = (1500, -600)
                elif window == 'tissue':
                    width, level = (400, 40)
                else:
                    raise ValueError(f"Window '{window}' not recognised.")
            else:
                width, level = window
            vmin = level - (width / 2)
            vmax = level + (width / 2)

            # Check that CT data isn't going to be hidden.
            ct_min, ct_max = ct_data.min(), ct_data.max()
            ct_width = ct_max - ct_min
            f = 0.1
            if ct_width < f * width:
                logging.warning(f"CT data range ({ct_min}, {ct_max}) is less than {f} * window range ({vmin}, {vmax}). You may be looking at grey - use a custom window (level, width).")
        else:
            vmin = ct_data.min()
            vmax = ct_data.max()
    else:
        vmin = 0
        vmax = 0

    # Plot CT data.
    if show_ct:
        ax.imshow(ct_slice, cmap='gray', aspect=aspect, interpolation='none', origin=__get_mpl_origin(view), vmin=vmin, vmax=vmax)
    else:
        # Plot black background.
        ax.imshow(np.zeros_like(ct_slice), cmap='gray', aspect=aspect, interpolation='none', origin=__get_mpl_origin(view))

    if not show_axes:
        ax.set_axis_off()

    if show_x_label:
        # Add 'x-axis' label.
        if view == 0:
            spacing_x = spacing[1]
        elif view == 1: 
            spacing_x = spacing[0]
        elif view == 2:
            spacing_x = spacing[0]

        ax.set_xlabel(f'voxel [@ {spacing_x:.3f} mm]')

    if show_y_label:
        # Add 'y-axis' label.
        if view == 0:
            spacing_y = spacing[2]
        elif view == 1:
            spacing_y = spacing[2]
        elif view == 2:
            spacing_y = spacing[1]
        ax.set_ylabel(f'voxel [@ {spacing_y:.3f} mm]')

    if region_data is not None:
        # Plot regions.
        okwargs = dict(
            alpha=alpha_region,
            colours=colours,
            crop=crop,
            escape_latex=escape_latex,
            legend_show_all_regions=legend_show_all_regions,
            show_extent=show_extent,
            view=view,
        )
        should_show_legend = __plot_region_data(region_data, ax, idx, aspect, **okwargs)

        # Create legend.
        if show_legend and should_show_legend:
            plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
            frame = plt_legend.get_frame()
            frame.set_boxstyle('square', pad=0.1)
            frame.set_edgecolor('black')
            frame.set_linewidth(linewidth)
            for l in plt_legend.get_lines():
                l.set_linewidth(linewidth_legend)

    # Plot landmarks.
    if landmark_data is not None:
        __plot_landmark_data(landmark_data, ax, fontsize, idx, size, view, **kwargs)

    # Set axis limits if cropped.
    if crop is not None:
        # Get new x ticks/labels.
        x_diff = int(np.diff(ax.get_xticks())[0])
        x_crop_max = crop[1][0] if crop[1][0] is not None else ct_slice.shape[1]
        x_tick_label_max = int(np.floor(x_crop_max / x_diff) * x_diff)
        x_crop_min = crop[0][0] if crop[0][0] is not None else 0
        x_tick_labels = list(range(x_crop_min, x_tick_label_max, x_diff))
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
        y_crop_max = crop[1][1] if crop[1][1] is not None else ct_slice.shape[0]
        y_tick_label_max = int(np.floor(y_crop_max / y_diff) * y_diff)
        y_crop_min = crop[0][1] if crop[0][1] is not None else 0
        y_tick_labels = list(range(y_crop_min, y_tick_label_max, y_diff))
        y_ticks = list((i * y_diff for i in range(len(y_tick_labels))))

        # Round up to nearest 'y_diff'.
        y_tick_labels_tmp = y_tick_labels.copy()
        y_tick_labels = [int(np.ceil(y / y_diff) * y_diff) for y in y_tick_labels_tmp]
        y_tick_diff = list(np.array(y_tick_labels) - np.array(y_tick_labels_tmp))
        y_ticks = list(np.array(y_ticks) + np.array(y_tick_diff))

        # Set y ticks.
        ax.set_yticks(y_ticks, labels=y_tick_labels)

    # Plot dose data.
    if dose_data is not None:
        # Create colormap with varying alpha - so 0 Gray is transparent.
        mpl_cmap = plt.get_cmap(dose_cmap)
        cmap = mpl_cmap(np.arange(mpl_cmap.N))

        # Set alpha(0)=0, and then linear from dose_alpha_min to dose_alpha_max.
        cmap[0, -1] = 0 # '-1' is the alpha channel.
        slope = (dose_alpha_max - dose_alpha_min) / (mpl_cmap.N - 1)
        cmap[1:, -1] = np.linspace(dose_alpha_min, dose_alpha_max, mpl_cmap.N - 1)
        cmap = ListedColormap(cmap)

        axim = ax.imshow(dose_slice, aspect=aspect, cmap=cmap, origin=__get_mpl_origin(view))
        if show_dose_bar:
            cbar = plt.colorbar(axim, fraction=dose_colourbar_size, pad=dose_colourbar_pad)
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
            title = plot_id
            if show_title_idx:
                title = f"{title}, {idx}/{n_slices - 1}"
            if show_title_view:
                title = f"{title} ({__view_to_text(view)})"

        # Escape text if using latex.
        if escape_latex:
            title = __escape_latex(title)

        ax.set_title(title)

    # Save plot to disk.
    if savepath is not None:
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show_figure:
        plt.show()

        # Revert latex settings.
        if escape_latex:
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
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[str] = None,
    crop: Box2D = None,
    crop_margin: float = 100,
    ct_data: Optional[np.ndarray] = None,
    escape_latex: bool = False,
    extent_of: Optional[Extrema] = None,
    figsize: Tuple[int, int] = (12, 12),
    fontsize: float = DEFAULT_FONT_SIZE,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
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
    show_pred_extent: bool = True,
    show_pred: bool = True,
    show_seg_patch: bool = True,
    idx: Optional[int] = None,
    truncate_spine: bool = True,
    view: Axis = 0,
    **kwargs: dict) -> None:
    __assert_idx(centre, extent_of, idx)
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
    if escape_latex:
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

    if centre is not None:
        # Get 'slice' at centre of data.
        label = region_data[centre] if type(centre) == str else centre
        extent_centre = centre_of_extent(label)
        idx = extent_centre[view]

    if extent_of is not None:
        if len(extent_of) == 2:
            eo_region, eo_end = extent_of
            eo_axis = view
        elif len(extent_of) == 3:
            eo_region, eo_end, eo_axis = extent_of

        # Get 'slice' at min/max extent of data.
        label = region_data[eo_region] if type(eo_region) == str else eo_region     # 'eo_region' can be str ('region_data' key) or np.ndarray.
        assert eo_end in ('min', 'max'), "'extent_of' must have one of ('min', 'max') as second element."
        eo_end = 0 if eo_end == 'min' else 1
        if postproc:
            label = postproc(label)
        ext = extent(label)
        idx = ext[eo_end][axis]

    # Plot patient regions.
    plot_patients_matrix(id, pred_data.shape, spacing, aspect=aspect, ax=ax, crop=crop, ct_data=ct_data, figsize=figsize, escape_latex=escape_latex, legend_loc=legend_loc, region_data=region_data, show_legend=show_legend, show_extent=show_label_extent, idx=idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'Box2D' type.
        if type(crop) == str:
            crop = __get_region_crop(region_data[crop], crop_margin, spacing, view)     # Crop was 'region_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'Box2D' type.

    # Plot prediction.
    if show_pred and not empty_pred:
        # Get aspect ratio.
        if not aspect:
            aspect = __get_mpl_aspect(view, spacing) 

        # Get slice data.
        pred_slice = __get_mpl_slice(pred_data, idx, view)

        # Crop the image.
        if crop:
            pred_slice = crop(pred_slice, __reverse_box_coords_2D(crop))

        # Plot prediction.
        colours = [(1, 1, 1, 0), pred_colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_mpl_origin(view))
        plt.plot(0, 0, c=pred_colour, label='Loc. Prediction')
        plt.contour(pred_slice, colors=[pred_colour], levels=[.5], linestyles='solid')

    # Plot prediction extent.
    if show_pred_extent and not empty_pred:
        # Get extent of prediction.
        pred_extent = extent(pred_data)

        # Plot extent if in view.
        label = 'Loc. extent' if __box_in_plane(pred_extent, view, idx) else 'Loc. extent (offscreen)'
        __plot_box_slice(pred_extent, view, colour=pred_extent_colour, crop=crop, label=label, linestyle='dashed')

    # Plot localiser centre.
    if show_pred_centre and not empty_pred:
        # Truncate if necessary to show true pred centre.
        centre_data = truncate(pred_data, spacing) if truncate_spine and pred_region == 'SpinalCord' else pred_data

        # Get pred centre.
        pred_centre = centre_of_extent(centre_data) 

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
        pred_centre = centre_of_extent(centre_data) 

        # Get second-stage patch.
        size = get_region_patch_size(pred_region, spacing)
        min, max = get_box(pred_centre, size)

        # Squash min/max to label size.
        min = np.clip(min, a_min=0, a_max=None)
        max = np.clip(max, a_min=None, a_max=pred_data.shape)

        if __box_in_plane((min, max), view, idx):
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
        if escape_latex:
            plt.rcParams.update({
                "font.family": rc_params['font.family'],
                'text.usetex': rc_params['text.usetex']
            })

    if close_figure:
        plt.close() 

def __get_region_centre(data: RegionLabel) -> int:
    if data.sum() == 0:
        raise ValueError("Centre region has no foreground voxels.")
    return centre_of_extent(data)

def __get_landmark_crop_box(
    landmark: PatientLandmark,
    crop_margin: float,
    size: ImageSize3D,
    spacing: ImageSpacing3D,
    view: Axis) -> Box2D:
    # Add crop margin.
    landmark = landmark[list(range(3))].iloc[0].to_numpy()
    crop_margin_vox = tuple(np.ceil(np.array(crop_margin) / spacing).astype(int))
    min = tuple(landmark - crop_margin_vox)
    max = tuple(landmark + crop_margin_vox)

    # Don't pad original image.
    min = tuple(np.clip(min, a_min=0, a_max=None))
    max = tuple(np.clip(max, a_min=None, a_max=size))

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

def __get_region_crop_box(
    data: np.ndarray,
    crop_margin: float,
    spacing: ImageSpacing3D,
    view: Axis) -> Box2D:
    # Get 3D crop box.
    ext = extent(data)

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

def plot_segmenter_predictions(
    id: str,
    spacing: ImageSpacing3D,
    pred_data: RegionLabels,
    alpha_diff: float = 0.7,
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: float = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    ct_data: Optional[np.ndarray] = None,
    escape_latex: bool = False,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[float, float] = (36, 12),
    fontsize: float = DEFAULT_FONT_SIZE,
    idx: Optional[int] = None,
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    pred_colours: Optional[Union[str, List[str]]] = None,
    region_data: Optional[RegionLabels] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_legend: bool = True,
    show_pred: bool = True,
    show_pred_extent: bool = False,
    view: Axis = 0,
    **kwargs: dict) -> None:
    __assert_idx(centre, extent_of, idx)
    pred_regions = tuple(pred_data.keys())
    n_pred_regions = len(pred_regions)
    n_regions = len(region_data.keys()) if region_data is not None else 0

    # Create plot figure/axis.
    if ax is None:
        _, axs = plt.subplots(1, 3, figsize=figsize)
    else:
        show = False

    # Get pred colours.
    if pred_colours is None:
        pred_colours = sns.color_palette('colorblind', n_regions)
    else:
        pred_colours = arg_to_list(colour, (str, tuple))

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if escape_latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Print prediction summary info.
    for r, pred in pred_data.items():
        if pred.sum() != 0:
            volume_vox = pred.sum()
            volume_mm3 = volume_vox * np.prod(spacing)
            logging.info(f"{r}: volume (vox.)={volume_vox}, volume (mm3)={volume_mm3}.")
        else:
            logging.info(f"{r}: empty.")

    # Get idx at centre of label.
    if centre is not None:
        label = region_data[centre] if type(centre) == str else centre
        extent_centre = centre_of_extent(label)
        idx = extent_centre[view]

    # Get idx at min/max extent of data.
    if extent_of is not None:
        label = region_data[extent_of[0]] if type(extent_of[0]) == str else extent_of
        extent_end = 0 if extent_of[1] == 'min' else 1
        ext = extent(label)
        idx = ext[extent_end][view]

    # Convert float idx to int.
    size = pred_data[list(pred_data.keys())[0]].shape
    if idx > 0 and idx < 1:
        idx = int(np.floor(idx * size[view]))

    # Plot patient regions - even if no 'ct_data/region_data' we still want to plot shape as black background.
    size = pred_data[list(pred_data.keys())[0]].shape
    okwargs = dict(
        alpha_region=alpha_region,
        aspect=aspect,
        ax=axs[0],
        colour=pred_colours,
        crops=crop,
        ct_datas=ct_data,
        escape_latex=escape_latex,
        legend_loc=legend_loc,
        region_datas=region_data,
        show=False,
        show_legend=False,
        idx=idx,
        views=view,
    )
    plot_patients_matrix(id, spacing, **okwargs, **kwargs)

    # Plot predictions.
    okwargs = dict(
        aspect=aspect,
        ax=axs[1],
        crops=crop,
        ct_datas=ct_data,
        escape_latex=escape_latex,
        legend_loc=legend_loc,
        show=False,
        show_legend=False,
        idx=idx,
        views=view,
    )
    plot_patients_matrix(id, spacing, **okwargs, **kwargs)

    for i in range(n_pred_regions):
        region = pred_regions[i]
        pred = pred_data[region]
        colour = pred_colours[i]

        if pred.sum() != 0 and show_pred:
            # Get aspect ratio.
            if not aspect:
                aspect = __get_mpl_aspect(view, spacing) 

            # Get slice data.
            pred_slice = __get_mpl_slice(pred, idx, view)

            # Crop the image.
            if crop:
                pred_slice = crop(pred_slice, __reverse_box_coords_2D(crop))

            # Plot prediction.
            if pred_slice.sum() != 0: 
                cmap = ListedColormap(((1, 1, 1, 0), colour))
                axs[1].imshow(pred_slice, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_mpl_origin(view))
                axs[1].plot(0, 0, c=colour, label=region)
                axs[1].contour(pred_slice, colors=[colour], levels=[.5], linestyles='solid')

        # Plot prediction extent.
        if pred.sum() != 0 and show_pred_extent:
            # Get prediction extent.
            pred_extent = extent(pred)

            # Plot if extent box is in view.
            label = f'{region} extent' if __box_in_plane(pred_extent, view, idx) else f'{region} extent (offscreen)'
            __plot_box_slice(pred_extent, view, ax=axs[1], colour=colour, crop=crop, label=label, linestyle='dashed')

    # Plot diff.
    okwargs = dict(
        aspect=aspect,
        ax=axs[2],
        crops=crop,
        ct_datas=ct_data,
        escape_latex=escape_latex,
        legend_loc=legend_loc,
        show=False,
        show_legend=False,
        idx=idx,
        views=view,
    )
    plot_patients_matrix(id, spacing, **okwargs, **kwargs)

    # Plot diffs.
    # Calculate single diff across all regions.
    label = np.zeros_like(region_data[list(region_data.keys())[0]])
    pred = np.zeros_like(pred_data[list(pred_data.keys())[0]])
    for i in range(n_pred_regions):
        region = pred_regions[i]
        label += region_data[region]
        label = np.clip(label, a_min=0, a_max=1)  # In case over overlapping regions.
        pred += pred_data[region]
        pred = np.clip(pred, a_min=0, a_max=1)  # In case over overlapping regions.
    diff = pred - label

    if diff.sum() != 0:
        # Get aspect ratio.
        if not aspect:
            aspect = __get_mpl_aspect(view, spacing) 

        # Get slice data.
        diff_slice = __get_mpl_slice(diff, idx, view)

        # Crop the image.
        if crop:
            diff_slice = crop(diff_slice, __reverse_box_coords_2D(crop))

        # Plot prediction.
        cmap = ListedColormap(('red', (1, 1, 1, 0), 'green'))     # Red for false negatives, green for false positives.
        axs[2].imshow(diff_slice, alpha=alpha_diff, aspect=aspect, cmap=cmap, origin=__get_mpl_origin(view))
        axs[2].plot(0, 0, c=colour, label=region)
        axs[2].contour(diff_slice, colors=[colour], levels=[.5], linestyles='solid')

    # Show legend.
    if show_legend:
        plt_legend = axs[0].legend(bbox_to_anchor=legend_bbox, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)
        plt_legend = axs[1].legend(bbox_to_anchor=legend_bbox, fontsize=fontsize, loc=legend_loc)
        for l in plt_legend.get_lines():
            l.set_linewidth(8)
        plt_legend = axs[2].legend(bbox_to_anchor=legend_bbox, fontsize=fontsize, loc=legend_loc)
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
    if escape_latex:
        plt.rcParams.update({
            "font.family": rc_params['font.family'],
            'text.usetex': rc_params['text.usetex']
        })

def plot_dataframe(
    ax: Optional[mpl.axes.Axes] = None,
    data: Optional[DataFrame] = None,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    dpi: float = 300,
    exclude_x: Optional[Union[str, List[str]]] = None,
    figsize_row: Tuple[float, float] = (12, 6),
    filt: Optional[Dict[str, Any]] = {},
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
    hspace: Optional[float] = None,
    include_x: Optional[Union[str, List[str]]] = None,
    legend_bbox: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
    legend_borderpad: float = 0.1,
    legend_loc: str = 'upper right',
    linecolour: str = 'black',
    linewidth: float = 0.5,
    major_tick_freq: Optional[float] = None,
    minor_tick_freq: Optional[float] = None,
    n_cols: Optional[int] = None,
    n_rows: Optional[int] = None,
    outlier_legend_loc: str = 'upper left',
    palette: Optional[_ColorPalette] = None,
    pointsize: float = 10,
    savepath: Optional[str] = None,
    save_pad_inches: float = 0.03,
    share_y: bool = True,
    show_boxes: bool = True,
    show_hue_connections: bool = False,
    show_hue_connections_inliers: bool = False,
    show_legend: Union[bool, List[bool]] = True,
    show_points: bool = True,
    show_stats: bool = False,
    show_x_tick_labels: bool = True,
    show_x_tick_label_counts: bool = True,
    stats_alt: AltHyp = AltHyp.TWO_SIDED,
    stats_bar_alg_use_lowest_level: bool = True,
    stats_bar_alpha: float = 0.5,
    # Stats bars must sit at least this high above data points.
    stats_bar_grid_offset: float = 0.015,           # Proportion of window height.
    # This value is important! Without this number, the first grid line
    # will only have one bar, which means our bars will be stacked much higher. This allows
    # a little wiggle room.
    stats_bar_grid_offset_wiggle: float = 0.01,     # Proportion of window height.
    stats_bar_grid_spacing: float = 0.04,          # Proportion of window height.
    stats_bar_height: float = 0.007,                # Proportion of window height.
    stats_bar_show_direction: bool = False,
    stats_bar_text_offset: float = 0.008,            # Proportion of window height.
    stats_boot_df: Optional[DataFrame] = None,
    stats_boot_df_cols: Optional[List[str]] = None,
    stats_exclude: List[str] = [],
    stats_exclude_left: List[str] = [],
    stats_exclude_right: List[str] = [],
    stats_index: Optional[Union[str, List[str]]] = None,
    stats_paired: bool = True,
    stats_sig: List[float] = [0.05, 0.01, 0.001],
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
    __assert_dataframe(data)
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

    # Filter data.
    for k, v in filt.items():
        data = data[data[k] == v]
        
    # Include/exclude.
    if include_xs is not None:
        data = data[data[x].isin(include_xs)]
    if exclude_xs is not None:
        data = data[~data[x].isin(exclude_xs)]

    # Add outlier data.
    data = __add_outlier_info(data, x, y, hue)

    # Calculate global min/max values for when sharing y-limits across
    # rows (share_y=True).
    global_min_y = data[y].min()
    global_max_y = data[y].max()

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
    if n_rows is None:
        n_rows = int(np.ceil(len(x_order) / n_cols))
    if ax is not None:
        assert n_rows == 1
        axs = [ax]
        # Figsize will have been handled externally.
    else:
        if n_rows > 1:
            gridspec_kw = { 'hspace': hspace }
            _, axs = plt.subplots(n_rows, 1, constrained_layout=True, dpi=dpi, figsize=(figsize_row[0], n_rows * figsize_row[1]), gridspec_kw=gridspec_kw, sharex=True, sharey=share_y)
        else:
            plt.figure(dpi=dpi, figsize=figsize_row)
            axs = [plt.gca()]

    # Get hue order/colour/labels.
    if hue is not None:
        if hue_order is None:
            hue_order = list(sorted(data[hue].unique()))

        # Calculate x width for each hue.
        hue_width = x_width / len(hue_order) 

        # Check there are enough colours in palette.
        if palette is None:
            palette = sns.color_palette('colorblind', len(hue_order))
        if len(hue_order) > len(palette):
            raise ValueError(f"'palette' doesn't have enough colours for hues '{hue_order}', needs '{len(hue_order)}'.")

        # Create map from hue to colour.
        hue_colours = dict((h, palette[i]) for i, h in enumerate(hue_order))

        if hue_labels is not None:
            if len(hue_labels) != len(hue_order):
                raise ValueError(f"Length of 'hue_labels' ({hue_labels}) should match hues ({hue_order}).")
    
    # Expand args to match number of rows.
    if isinstance(show_legend, bool):
        show_legends = [show_legend] * n_rows
    else: 
        if len(show_legend) != n_rows:
            raise ValueError(f"Length of 'show_legend' ({len(show_legend)}) should match number of rows ({n_rows}).")
        else:
            show_legends = show_legend
    if legend_bbox is None or isinstance(legend_bbox, tuple):
        legend_bboxs = [legend_bbox] * n_rows
    else: 
        if len(legend_bbox) != n_rows:
            raise ValueError(f"Length of 'legend_bbox' ({len(legend_bbox)}) should match number of rows ({n_rows}).")
        else:
            legend_bboxs = legend_bbox

    # Plot rows.
    for i, show_legend, legend_bbox in zip(range(n_rows), show_legends, legend_bboxs):
        # Split data.
        row_x_order = x_order[i * n_cols:(i + 1) * n_cols]
        row_x_tick_labels = x_tick_labels[i * n_cols:(i + 1) * n_cols]

        # Get x colours.
        if hue is None:
            # Check there are enough colors in palette.
            if palette is None:
                palette = sns.color_palette('colorblind', len(row_x_order))
            if len(row_x_order) > len(palette):
                raise ValueError(f"'palette' doesn't have enough colours for x values '{row_x_order}', needs '{len(row_x_order)}'.")
            x_colours = dict((x, palette[i]) for i, x in enumerate(row_x_order))

        # Get row data.
        row_df = data[data[x].isin(row_x_order)].copy()

        # Get x-axis limits.
        x_lim_row = list(x_lim)
        n_cols_row = len(row_x_order)
        if x_lim_row[0] is None:
            x_lim_row[0] = -0.5
        if x_lim_row[1] is None:
            x_lim_row[1] = n_cols_row - 0.5

        # Get y-axis limits.
        y_margin = 0.05
        row_min_y = row_df[y].min()
        row_max_y = row_df[y].max()
        min_y = global_min_y if share_y else row_min_y
        max_y = global_max_y if share_y else row_max_y
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

        # Set axis limits.
        # This has to be done twice - once to set parent axes, and once to set child (inset) axes.
        # I can't remember why we made inset axes...?
        # Make this a function, as we need to call when stats bars exceed the y-axis limit.
        inset_ax = __set_axes_limits(axs[i], x_lim_row, y_lim_row)

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
                            res = inset_ax.boxplot(hue_df[y].dropna(), boxprops=dict(color=linecolour, facecolor=hue_colours[hue_name], linewidth=linewidth), capprops=dict(color=linecolour, linewidth=linewidth), flierprops=dict(color=linecolour, linewidth=linewidth, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=linewidth), patch_artist=True, positions=[hue_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=linewidth), widths=hue_width)
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
                            res = inset_ax.violinplot(hue_df[y], positions=[hue_pos], widths=hue_width)

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
                        inset_ax.boxplot(x_df[y], boxprops=dict(color=linecolour, facecolor=x_colours[x_val], linewidth=linewidth), capprops=dict(color=linecolour, linewidth=linewidth), flierprops=dict(color=linecolour, linewidth=linewidth, marker='D', markeredgecolor=linecolour), medianprops=dict(color=linecolour, linewidth=linewidth), patch_artist=True, positions=[x_pos], showfliers=False, whiskerprops=dict(color=linecolour, linewidth=linewidth))
                    elif style == 'violin':
                        inset_ax.violinplot(x_df[y], positions=[x_pos])

            # Plot points.
            if show_points:
                if hue is not None:
                    for j, hue_name in enumerate(hue_order):
                        hue_df = row_df[(row_df[x] == x_val) & (row_df[hue] == hue_name)]
                        if len(hue_df) == 0:
                            continue
                        res = inset_ax.scatter(hue_df['x_pos'], hue_df[y], color=hue_colours[hue_name], edgecolors=linecolour, linewidth=linewidth, s=pointsize, zorder=100)
                        if not hue_label in hue_artists:
                            hue_artists[hue_label] = res
                else:
                    x_df = row_df[row_df[x] == x_val]
                    inset_ax.scatter(x_df['x_pos'], x_df[y], color=x_colours[x_val], edgecolors=linecolour, linewidth=linewidth, s=pointsize, zorder=100)

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
                    lines = inset_ax.plot(x_pos, y_data, color=line_palette[j])

                    # Save line/label for legend.
                    artists.append(lines[0])
                    label = ':'.join(line_id.tolist())
                    labels.append(label)

                # Annotate outlier legend.
                if show_legend:
                    # Save main legend.
                    main_legend = inset_ax.get_legend()

                    # Show outlier legend.
                    inset_ax.legend(artists, labels, borderpad=legend_borderpad, bbox_to_anchor=legend_bbox, fontsize=fontsize_legend, loc=outlier_legend_loc)

                    # Re-add main legend.
                    inset_ax.add_artist(main_legend)

        # Show legend.
        if hue is not None:
            if show_legend:
                # Filter 'hue_labels' based on hue 'artists'. Some hues may not be present in this row,
                # and 'hue_labels' is a global (across all rows) tracker.
                hue_labels = hue_order if hue_labels is None else hue_labels
                labels, artists = list(zip(*[(h, hue_artists[h]) for h in hue_labels if h in hue_artists]))

                # Show legend.
                legend = inset_ax.legend(artists, labels, borderpad=legend_borderpad, bbox_to_anchor=legend_bbox, fontsize=fontsize_legend, loc=legend_loc)
                frame = legend.get_frame()
                frame.set_boxstyle('square')
                frame.set_edgecolor('black')
                frame.set_linewidth(linewidth)

        # Get pairs for stats tests.
        if show_stats:
            if hue is None:
                # Create pairs of 'x' values.
                if n_rows != 1:
                    raise ValueError(f"Can't show stats between 'x' values with multiple rows - not handled.")

                pairs = []
                max_skips = len(x_order) - 1
                for skip in range(1, max_skips + 1):
                    for j, x_val in enumerate(x_order):
                        other_x_idx = j + skip
                        if other_x_idx < len(x_order):
                            pair = (x_val, x_order[other_x_idx])
                            pairs.append(pair)
            else:
                # Create pairs of 'hue' values.
                pairs = []
                for x_val in row_x_order:
                    # Create pairs - start at lower numbers of skips as this will result in a 
                    # condensed plot.
                    hue_pairs = []
                    max_skips = len(hue_order) - 1
                    for skip in range(1, max_skips + 1):
                        for j, hue_val in enumerate(hue_order):
                            other_hue_index = j + skip
                            if other_hue_index < len(hue_order):
                                pair = (hue_val, hue_order[other_hue_index])
                                hue_pairs.append(pair)
                    pairs.append(hue_pairs)

        # Get p-values for each pair.
        if show_stats:
            nonsig_pairs = []
            nonsig_p_vals = []
            sig_pairs = []
            sig_p_vals = []

            if hue is None:
                for x_l, x_r in pairs:
                    row_pivot_df = row_df.pivot(index=stats_index, columns=x, values=y).reset_index()
                    if x_l in row_pivot_df.columns and x_r in row_pivot_df.columns:
                        vals_l = row_pivot_df[x_l]
                        vals_r = row_pivot_df[x_r]

                        p_val = __calculate_p_val(vals_l, vals_r, stats_alt, stats_paired, stats_sig)

                        if p_val < stats_sig[0]:
                            sig_pairs.append((x_l, x_r))
                            sig_p_vals.append(p_val)
                        else:
                            nonsig_pairs.append((x_l, x_r))
                            nonsig_p_vals.append(p_val)
            else:
                for x_val, hue_pairs in zip(row_x_order, pairs):
                    x_df = row_df[row_df[x] == x_val]

                    hue_nonsig_pairs = []
                    hue_nonsig_p_vals = []
                    hue_sig_pairs = []
                    hue_sig_p_vals = []
                    for hue_l, hue_r in hue_pairs:
                        if stats_boot_df is not None:
                            # Load p-values from 'stats_boot_df'.
                            x_pivot_df = x_df.pivot(index=stats_index, columns=[hue], values=[y]).reset_index()
                            if (y, hue_l) in x_pivot_df.columns and (y, hue_r) in x_pivot_df.columns:
                                vals_l = x_pivot_df[y][hue_l]
                                vals_r = x_pivot_df[y][hue_r]

                                # Don't add stats pair if main data is empty.
                                if len(vals_l) == 0 or len(vals_r) == 0:
                                    continue
                            else:
                                # Don't add stats pair if main data is empty.
                                continue
                            
                            # Get ('*', '<direction>') from dataframe. We have 'x_val' which is our region.
                            x_boot_df = stats_boot_df[stats_boot_df[x] == x_val]
                            boot_hue_l, boot_hue_r, boot_p_val = stats_boot_df_cols
                            x_pair_boot_df = x_boot_df[(x_boot_df[boot_hue_l] == hue_l) & (x_boot_df[boot_hue_r] == hue_r)]
                            if len(x_pair_boot_df) == 0:
                                raise ValueError(f"No matches found in 'stats_boot_df' for '{x}' ('{x_val}') '{boot_hue_l}' ('{hue_l}') and '{boot_hue_r}' ('{hue_r}').")
                            if len(x_pair_boot_df) > 1:
                                raise ValueError(f"Found multiple matches in 'stats_boot_df' for '{x}' ('{x_val}') '{boot_hue_l}' ('{hue_l}') and '{boot_hue_r}' ('{hue_r}').")
                            p_val = x_pair_boot_df.iloc[0][boot_p_val]

                            if p_val != '':
                                hue_sig_pairs.append((hue_l, hue_r))
                                hue_sig_p_vals.append(p_val)
                            else:
                                hue_nonsig_pairs.append((hue_l, hue_r))
                                hue_nonsig_p_vals.append(p_val)
                    else:
                        # Calculate p-values using stats tests.
                        for hue_l, hue_r in hue_pairs:
                            x_pivot_df = x_df.pivot(index=stats_index, columns=[hue], values=[y]).reset_index()
                            if (y, hue_l) in x_pivot_df.columns and (y, hue_r) in x_pivot_df.columns:
                                vals_l = x_pivot_df[y][hue_l]
                                vals_r = x_pivot_df[y][hue_r]

                                p_val = __calculate_p_val(vals_l, vals_r, stats_alt, stats_paired, stats_sig)

                                if p_val < stats_sig[0]:
                                    hue_sig_pairs.append((hue_l, hue_r))
                                    hue_sig_p_vals.append(p_val)
                                else:
                                    hue_nonsig_pairs.append((hue_l, hue_r))
                                    hue_nonsig_p_vals.append(p_val)
                
                    nonsig_pairs.append(hue_nonsig_pairs)
                    nonsig_p_vals.append(hue_nonsig_p_vals)
                    sig_pairs.append(hue_sig_pairs)
                    sig_p_vals.append(hue_sig_p_vals)

        # Format p-values.
        if show_stats:
            if hue is None:
                sig_p_vals = __format_p_vals(sig_p_vals, stats_sig)
            else:
                sig_p_vals = [__format_p_vals(p, stats_sig) for p in sig_p_vals]

        # Remove 'excluded' pairs.
        if show_stats:
            filt_pairs = []
            filt_p_vals = []
            if hue is None:
                for (x_l, x_r), p_val in zip(sig_pairs, sig_p_vals):
                    if (x_l in stats_exclude or x_r in stats_exclude) or (x_l in stats_exclude_left) or (x_r in stats_exclude_right):
                        continue
                    filt_pairs.append((x_l, x_r))
                    filt_p_vals.append(p_val)
            else:
                for ps, p_vals in zip(sig_pairs, sig_p_vals):
                    hue_filt_pairs = []
                    hue_filt_p_vals = []
                    for (hue_l, hue_r), p_val in zip(ps, p_vals):
                        if (hue_l in stats_exclude or hue_r in stats_exclude) or (hue_l in stats_exclude_left) or (hue_r in stats_exclude_right):
                            continue
                        hue_filt_pairs.append((hue_l, hue_r))
                        hue_filt_p_vals.append(p_val)
                    filt_pairs.append(hue_filt_pairs)
                    filt_p_vals.append(hue_filt_p_vals)

        # Display stats bars.
        # To display stats bars, we fit a vertical grid over the plot and place stats bars
        # on the grid lines - so they look nice.
        if show_stats:
            # Calculate heights based on window height.
            y_height = max_y - min_y
            stats_bar_height = y_height * stats_bar_height
            stats_bar_grid_spacing = y_height * stats_bar_grid_spacing
            stats_bar_grid_offset = y_height * stats_bar_grid_offset
            stats_bar_grid_offset_wiggle = y_height * stats_bar_grid_offset_wiggle
            stats_bar_text_offset = y_height * stats_bar_text_offset
                
            if hue is None:
                # Calculate 'y_grid_offset' - the bottom of the grid.
                # For each pair, we calculate the max value of the data, as the bar should
                # lie above this. Then we find the smallest of these max values across all pairs.
                # 'stats_bar_grid_offset' is added to give spacing between the data and the bar.
                y_grid_offset = np.inf
                min_skip = None
                for x_l, x_r in filt_pairs:
                    if stats_bar_alg_use_lowest_level:
                        skip = x_order.index(x_r) - x_order.index(x_l) - 1
                        if min_skip is None:
                            min_skip = skip
                        elif skip > min_skip:
                            continue

                    x_l_df = row_df[row_df[x] == x_l]
                    x_r_df = row_df[row_df[x] == x_r]
                    y_max = max(x_l_df[y].max(), x_r_df[y].max())
                    y_max = y_max + stats_bar_grid_offset
                    if y_max < y_grid_offset:
                        y_grid_offset = y_max

                # Add data offset.
                y_grid_offset = y_grid_offset + stats_bar_grid_offset_wiggle

                # Annotate figure.
                # We keep track of bars we've plotted using 'y_idxs'.
                # This is a mapping from the hue to the grid positions that have already
                # been used for either a left or right hand side of a bar.
                y_idxs: Dict[str, List[int]] = {}
                for (x_l, x_r), p_val in zip(filt_pairs, filt_p_vals):
                    # Get plot 'x_pos' for each x value.
                    x_l_df = row_df[row_df[x] == x_l]
                    x_r_df = row_df[row_df[x] == x_r]
                    x_left = x_l_df['x_pos'].iloc[0]
                    x_right = x_r_df['x_pos'].iloc[0]

                    # Get 'y_idx_min' (e.g. 0, 1, 2,...) which tells us the lowest grid line
                    # we can use based on our data points.
                    # We calculate this by finding the max data value for the pair, and also
                    # the max values for any hues between the pair values - as our bar should
                    # not collide with these 'middle' hues. 
                    y_data_maxes = [x_end_df[y].max() for x_end_df in [x_l_df, x_r_df]]
                    n_mid_xs = x_order.index(x_r) - x_order.index(x_l) - 1
                    for j in range(n_mid_xs):
                        x_mid = x_order[x_order.index(x_l) + j + 1]
                        x_mid_df = row_df[row_df[x] == x_mid]
                        y_data_max = x_mid_df[y].max()
                        y_data_maxes.append(y_data_max)
                    y_data_max = max(y_data_maxes) + stats_bar_grid_offset
                    y_idx_min = int(np.ceil((y_data_max - y_grid_offset) / stats_bar_grid_spacing))

                    # We don't want our new stats bar to collide with any existing bars.
                    # Get the y positions for all stats bar that have already been plotted
                    # and that have their left or right end at one of the 'middle' hues for
                    # our current pair.
                    n_mid_xs = x_order.index(x_r) - x_order.index(x_l) - 1
                    y_idxs_mid_xs = []
                    for j in range(n_mid_xs):
                        x_mid = x_order[x_order.index(x_l) + j + 1]
                        if x_mid in y_idxs:
                            y_idxs_mid_xs += y_idxs[x_mid]

                    # Get the next free position that doesn't collide with any existing bars.
                    y_idx_max = 100
                    for y_idx in range(y_idx_min, y_idx_max):
                        if y_idx not in y_idxs_mid_xs:
                            break

                    # Plot bar.
                    y_min = y_grid_offset + y_idx * stats_bar_grid_spacing
                    y_max = y_min + stats_bar_height
                    inset_ax.plot([x_left, x_left, x_right, x_right], [y_min, y_max, y_max, y_min], alpha=stats_bar_alpha, color=linecolour, linewidth=linewidth)    

                    # Adjust y-axis limits if bar would be plotted outside of window.
                    # Unless y_lim is set manually.
                    y_lim_top = y_max + 1.5 * stats_bar_grid_spacing
                    if y_lim[1] is None and y_lim_top > y_lim_row[1]:
                        y_lim_row[1] = y_lim_top
                        inset_ax = __set_axes_limits(axs[i], x_lim_row, y_lim_row, inset_ax=inset_ax)

                    # Plot p-value.
                    x_text = (x_left + x_right) / 2
                    y_text = y_max + stats_bar_text_offset
                    inset_ax.text(x_text, y_text, p_val, alpha=stats_bar_alpha, fontsize=fontsize_stats, horizontalalignment='center', verticalalignment='center')

                    # Save position of plotted stats bar.
                    if not x_l in y_idxs:
                        y_idxs[x_l] = [y_idx]
                    elif y_idx not in y_idxs[x_l]:
                        y_idxs[x_l] = list(sorted(y_idxs[x_l] + [y_idx]))
                    if not x_r in y_idxs:
                        y_idxs[x_r] = [y_idx]
                    elif y_idx not in y_idxs[x_r]:
                        y_idxs[x_r] = list(sorted(y_idxs[x_r] + [y_idx]))
            else:
                for hue_pairs, hue_p_vals in zip(filt_pairs, filt_p_vals):
                    # Calculate 'y_grid_offset' - the bottom of the grid.
                    # For each pair, we calculate the max value of the data, as the bar should
                    # lie above this. Then we find the smallest of these max values across all pairs.
                    # 'stats_bar_grid_offset' is added to give spacing between the data and the bar.
                    y_grid_offset = np.inf
                    min_skip = None
                    for hue_l, hue_r in hue_pairs:
                        if stats_bar_alg_use_lowest_level:
                            skip = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                            if min_skip is None:
                                min_skip = skip
                            elif skip > min_skip:
                                continue

                        hue_l_df = x_df[x_df[hue] == hue_l]
                        hue_r_df = x_df[x_df[hue] == hue_r]
                        y_max = max(hue_l_df[y].max(), hue_r_df[y].max())
                        y_max = y_max + stats_bar_grid_offset
                        if y_max < y_grid_offset:
                            y_grid_offset = y_max

                    # Add data offset.
                    y_grid_offset = y_grid_offset + stats_bar_grid_offset_wiggle

                    # Annotate figure.
                    # We keep track of bars we've plotted using 'y_idxs'.
                    # This is a mapping from the hue to the grid positions that have already
                    # been used for either a left or right hand side of a bar.
                    y_idxs: Dict[str, List[int]] = {}
                    for (hue_l, hue_r), p_val in zip(hue_pairs, hue_p_vals):
                        # Get plot 'x_pos' for each hue.
                        hue_l_df = x_df[x_df[hue] == hue_l]
                        hue_r_df = x_df[x_df[hue] == hue_r]
                        x_left = hue_l_df['x_pos'].iloc[0]
                        x_right = hue_r_df['x_pos'].iloc[0]

                        # Get 'y_idx_min' (e.g. 0, 1, 2,...) which tells us the lowest grid line
                        # we can use based on our data points.
                        # We calculate this by finding the max data value for the pair, and also
                        # the max values for any hues between the pair values - as our bar should
                        # not collide with these 'middle' hues. 
                        y_data_maxes = [hue_df[y].max() for hue_df in [hue_l_df, hue_r_df]]
                        n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                        for j in range(n_mid_hues):
                            hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                            hue_mid_df = x_df[x_df[hue] == hue_mid]
                            y_data_max = hue_mid_df[y].max()
                            y_data_maxes.append(y_data_max)
                        y_data_max = max(y_data_maxes) + stats_bar_grid_offset
                        y_idx_min = int(np.ceil((y_data_max - y_grid_offset) / stats_bar_grid_spacing))

                        # We don't want our new stats bar to collide with any existing bars.
                        # Get the y positions for all stats bar that have already been plotted
                        # and that have their left or right end at one of the 'middle' hues for
                        # our current pair.
                        n_mid_hues = hue_order.index(hue_r) - hue_order.index(hue_l) - 1
                        y_idxs_mid_hues = []
                        for j in range(n_mid_hues):
                            hue_mid = hue_order[hue_order.index(hue_l) + j + 1]
                            if hue_mid in y_idxs:
                                y_idxs_mid_hues += y_idxs[hue_mid]

                        # Get the next free position that doesn't collide with any existing bars.
                        y_idx_max = 100
                        for y_idx in range(y_idx_min, y_idx_max):
                            if y_idx not in y_idxs_mid_hues:
                                break

                        # Plot bar.
                        y_min = y_grid_offset + y_idx * stats_bar_grid_spacing
                        y_max = y_min + stats_bar_height
                        inset_ax.plot([x_left, x_left, x_right, x_right], [y_min, y_max, y_max, y_min], alpha=stats_bar_alpha, color=linecolour, linewidth=linewidth)    

                        # Adjust y-axis limits if bar would be plotted outside of window.
                        # Unless y_lim is set manually.
                        y_lim_top = y_max + 1.5 * stats_bar_grid_spacing
                        if y_lim[1] is None and y_lim_top > y_lim_row[1]:
                            y_lim_row[1] = y_lim_top
                            inset_ax = __set_axes_limits(axs[i], x_lim_row, y_lim_row, inset_ax=inset_ax)

                        # Plot p-value.
                        x_text = (x_left + x_right) / 2
                        y_text = y_max + stats_bar_text_offset
                        inset_ax.text(x_text, y_text, p_val, alpha=stats_bar_alpha, fontsize=fontsize_stats, horizontalalignment='center', verticalalignment='center')

                        # Save position of plotted stats bar.
                        if not hue_l in y_idxs:
                            y_idxs[hue_l] = [y_idx]
                        elif y_idx not in y_idxs[hue_l]:
                            y_idxs[hue_l] = list(sorted(y_idxs[hue_l] + [y_idx]))
                        if not hue_r in y_idxs:
                            y_idxs[hue_r] = [y_idx]
                        elif y_idx not in y_idxs[hue_r]:
                            y_idxs[hue_r] = list(sorted(y_idxs[hue_r] + [y_idx]))
          
        # Set axis labels.
        x_label = x_label if x_label is not None else ''
        y_label = y_label if y_label is not None else ''
        inset_ax.set_xlabel(x_label, fontsize=fontsize_label)
        inset_ax.set_ylabel(y_label, fontsize=fontsize_label)
                
        # Set axis tick labels.
        inset_ax.set_xticks(list(range(len(row_x_tick_labels))))
        if show_x_tick_labels:
            inset_ax.set_xticklabels(row_x_tick_labels, fontsize=fontsize_tick_label, rotation=x_tick_label_rot)
        else:
            inset_ax.set_xticklabels([])

        inset_ax.tick_params(axis='y', which='major', labelsize=fontsize_tick_label)

        # Set y axis major ticks.
        if major_tick_freq is not None:
            major_tick_min = y_lim[0]
            if major_tick_min is None:
                major_tick_min = inset_ax.get_ylim()[0]
            major_tick_max = y_lim[1]
            if major_tick_max is None:
                major_tick_max = inset_ax.get_ylim()[1]
            
            # Round range to nearest multiple of 'major_tick_freq'.
            major_tick_min = np.ceil(major_tick_min / major_tick_freq) * major_tick_freq
            major_tick_max = np.floor(major_tick_max / major_tick_freq) * major_tick_freq
            n_major_ticks = int((major_tick_max - major_tick_min) / major_tick_freq) + 1
            major_ticks = np.linspace(major_tick_min, major_tick_max, n_major_ticks)
            major_tick_labels = [str(round(t, 3)) for t in major_ticks]     # Some weird str() conversion without rounding.
            inset_ax.set_yticks(major_ticks)
            inset_ax.set_yticklabels(major_tick_labels)

        # Set y axis minor ticks.
        if minor_tick_freq is not None:
            minor_tick_min = y_lim[0]
            if minor_tick_min is None:
                minor_tick_min = inset_ax.get_ylim()[0]
            minor_tick_max = y_lim[1]
            if minor_tick_max is None:
                minor_tick_max = inset_ax.get_ylim()[1]
            
            # Round range to nearest multiple of 'minor_tick_freq'.
            minor_tick_min = np.ceil(minor_tick_min / minor_tick_freq) * minor_tick_freq
            minor_tick_max = np.floor(minor_tick_max / minor_tick_freq) * minor_tick_freq
            n_minor_ticks = int((minor_tick_max - minor_tick_min) / minor_tick_freq) + 1
            minor_ticks = np.linspace(minor_tick_min, minor_tick_max, n_minor_ticks)
            inset_ax.set_yticks(minor_ticks, minor=True)

        # Set y grid lines.
        inset_ax.grid(axis='y', alpha=0.1, color='grey', linewidth=linewidth)
        inset_ax.set_axisbelow(True)

        # Set axis spine/tick linewidths and tick lengths.
        spines = ['top', 'bottom','left','right']
        for spine in spines:
            inset_ax.spines[spine].set_linewidth(linewidth)
        inset_ax.tick_params(which='both', length=ticklength, width=linewidth)

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
        plt.savefig(savepath, bbox_inches='tight', dpi=dpi, pad_inches=save_pad_inches)
        logging.info(f"Saved plot to '{savepath}'.")

def plot_registration(
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    cca: bool = False,
    colour: Optional[Union[str, List[str]]] = None,
    dose_alpha_min: float = 0.1,
    dose_alpha_max: float = 0.7,
    dose_colourbar_size: float = 0.03,
    dose_data: Optional[np.ndarray] = None,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'region_data' if 'str', else 'np.ndarray') and min/max of extent.
    fixed_centre: Optional[Union[str, np.ndarray]] = None,             # Uses 'fixed_region_data' if 'str', else uses 'np.ndarray'.
    fixed_crop: Optional[Union[str, np.ndarray, Box2D]] = None,    # Uses 'fixed_region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    fixed_crop_margin: float = 100,                                       # Applied if cropping to 'fixed_region_data' or 'np.ndarray'
    fixed_ct_data: Optional[np.ndarray] = None,
    fixed_idx: Optional[int] = None,
    fixed_landmark_data: Optional[Landmarks] = None,
    fixed_offset: Optional[PointMM3D] = None,
    fixed_region_data: Optional[np.ndarray] = None,
    fixed_spacing: Optional[ImageSpacing3D] = None,
    figsize: Tuple[float, float] = (30, 10),
    fontsize: int = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_show_all_regions: bool = False,
    match_landmarks: bool = True,
    moved_centre: Optional[Union[str, np.ndarray]] = None,             # Uses 'moved_region_data' if 'str', else uses 'np.ndarray'.
    moved_colour: str = 'red',
    moved_crop: Optional[Union[str, np.ndarray, Box2D]] = None,    # Uses 'moved_region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moved_crop_margin: float = 100,                                       # Applied if cropping to 'moved_region_data' or 'np.ndarray'
    moved_ct_data: Optional[np.ndarray] = None,
    moved_idx: Optional[int] = None,
    moved_landmark_data: Optional[Landmarks] = None,
    moved_region_data: Optional[np.ndarray] = None,
    moved_use_fixed_crop: bool = True,
    moved_use_fixed_idx: bool = True,
    moving_centre: Optional[Union[str, np.ndarray]] = None,             # Uses 'moving_region_data' if 'str', else uses 'np.ndarray'.
    moving_crop: Optional[Union[str, np.ndarray, Box2D]] = None,    # Uses 'moving_region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moving_crop_margin: float = 100,                                       # Applied if cropping to 'moving_region_data' or 'np.ndarray'
    moving_ct_data: Optional[np.ndarray] = None,
    moving_idx: Optional[int] = None,
    moving_landmark_data: Optional[Landmarks] = None,
    moving_offset: Optional[PointMM3D] = None,
    moving_region_data: Optional[np.ndarray] = None,
    moving_spacing: Optional[ImageSpacing3D] = None,
    n_landmarks: Optional[int] = None,
    norm: Optional[Tuple[float, float]] = None,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
    show_axes: bool = True,
    show_extent: bool = True,
    show_fixed: bool = True,
    show_grid: bool = True,
    show_legend: bool = False,
    show_moved_landmarks: bool = True,
    show_moving: bool = True,
    show_moving_landmarks: bool = True,
    show_region_overlay: bool = True,
    show_title: bool = True,
    show_x_label: bool = True,
    show_x_ticks: bool = True,
    show_y_label: bool = True,
    show_y_ticks: bool = True,
    transform: Optional[Union[itk.Transform, sitk.Transform]] = None,
    transform_format: Literal['itk', 'sitk'] = 'sitk',
    title: Optional[str] = None,
    view: Axis = 0,
    window: Optional[Union[Literal['bone', 'lung', 'tissue'], Tuple[float, float]]] = None,
    **kwargs) -> None:
    __assert_idx(fixed_centre, extent_of, fixed_idx)
    __assert_idx(moving_centre, extent_of, moving_idx)

    if n_landmarks is not None and match_landmarks:
        # Ensure that the n "moving/moved" landmarks targeted by "n_landmarks" are
        # are the same as the n "fixed" landmarks. 

        # Get n fixed landmarks.
        fixed_idx = __convert_float_idx(fixed_idx, fixed_ct_data.shape, view)
        fixed_landmark_data['dist'] = np.abs(fixed_landmark_data[view] - fixed_idx)
        fixed_landmark_data = fixed_landmark_data.sort_values('dist')
        fixed_landmark_data = fixed_landmark_data.iloc[:n_landmarks]
        fixed_landmarks = fixed_landmark_data['landmark-id'].tolist()

        # Get moving/moved landmarks.
        moving_landmark_data = moving_landmark_data[moving_landmark_data['landmark-id'].isin(fixed_landmarks)]
        moved_landmark_data = moved_landmark_data[moved_landmark_data['landmark-id'].isin(fixed_landmarks)]

        okwargs = dict(
            colour=moved_colour,
            n_landmarks=n_landmarks, 
            zorder=0,  # Plot underneath "moving" (ground truth) landmarks.
        )

    # Get all plot parameters.
    hidden = 'HIDDEN'
    ct_datas = [moving_ct_data if show_moving else hidden, fixed_ct_data if show_fixed else hidden, moved_ct_data]
    ct_datas = [c for c in ct_datas if not (isinstance(c, str) and c == hidden)]
    spacings = [moving_spacing if show_moving else hidden, fixed_spacing if show_fixed else hidden, fixed_spacing]
    spacings = [c for c in spacings if not (isinstance(c, str) and c == hidden)]
    offsets = [moving_offset if show_moving else hidden, fixed_offset if show_fixed else hidden, fixed_offset]
    offsets = [c for c in offsets if not (isinstance(c, str) and c == hidden)]
    sizes = [moving_ct_data.shape if show_moving else hidden, fixed_ct_data.shape if show_fixed else hidden, moved_ct_data.shape]
    sizes = [c for c in sizes if not (isinstance(c, str) and c == hidden)]
    centres = [moving_centre if show_moving else hidden, fixed_centre if show_fixed else hidden, moved_centre]
    centres = [c for c in centres if not (isinstance(c, str) and c == hidden)]
    crops = [moving_crop if show_moving else hidden, fixed_crop if show_fixed else hidden, moved_crop]
    crops = [c for c in crops if not (isinstance(c, str) and c == hidden)]
    crop_margins = [moving_crop_margin if show_moving else hidden, fixed_crop_margin if show_fixed else hidden, moved_crop_margin]
    crop_margins = [c for c in crop_margins if not (isinstance(c, str) and c == hidden)]
    ids = [f'{moving_pat_id}:{moving_study_id}' if show_moving else hidden, f'{fixed_pat_id}:{fixed_study_id}' if show_fixed else hidden, f'{moving_pat_id}:{moving_study_id} -> {fixed_pat_id}:{fixed_study_id}']
    ids = [c for c in ids if not (isinstance(c, str) and c == hidden)]
    landmark_datas = [moving_landmark_data if show_moving else hidden, fixed_landmark_data if show_fixed else hidden, None]
    if not show_moving_landmarks:
        # Hide moving landmarks, but we still need 'moving_landmark_data' != None for the
        # "moved" landmark plotting code.
        landmark_datas[0] = None
    landmark_datas = [l for l in landmark_datas if not (isinstance(l, str) and l == hidden)]
    region_datas = [moving_region_data if show_moving else hidden, fixed_region_data if show_fixed else hidden, moved_region_data]
    region_datas = [c for c in region_datas if not (isinstance(c, str) and c == hidden)]
    idxs = [moving_idx if show_moving else hidden, fixed_idx if show_fixed else hidden, moved_idx]
    idxs = [c for c in idxs if not (isinstance(c, str) and c == hidden)]
    infos = [{} if show_moving else hidden, {} if show_fixed else hidden, { 'fixed': moved_use_fixed_idx }]
    infos = [c for c in infos if not (isinstance(c, str) and c == hidden)]

    n_figs = show_moving + show_fixed + 1
    axs = arg_to_list(ax, mpl.axes.Axes)
    if axs is None:
        n_rows = 2 if show_grid else 1
        n_cols = n_figs
        figsize_width, figsize_height = figsize
        figsize_height = n_rows * figsize_height
        figsize = (figsize_width, figsize_height)
        # How do I remove vertical spacing???
        fig = plt.figure(figsize=__convert_figsize_to_inches(figsize))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.05, wspace=0.3)
        axs = []
        # Add first row of images.
        for i in range(n_figs):
            ax = fig.add_subplot(gs[0, i])
            axs.append(ax)

        # Add second row of images.
        if show_grid or show_region_overlay:
            for i in range(n_figs):
                ax = fig.add_subplot(gs[1, i])
                axs.append(ax)

            # Remove axes from second row if necessary.
            if not show_region_overlay:
                axs[2 * n_figs - 2].remove()

        close_figure = True
    else:
        assert len(axs) == n_figs
        close_figure = False
        show = False

    # Set latex as text compiler.
    rc_params = plt.rcParams.copy()
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Plot images.
    for i in range(n_figs):
        okwargs = dict(
            ax=axs[i],
            centres=centres[i],
            crops=crops[i],
            ct_datas=ct_datas[i],
            idx=idxs[i],
            landmark_datas=landmark_datas[i],
            n_landmarks=n_landmarks,
            region_datas=region_datas[i],
            show_legend=show_legend,
            views=view,
        )
        plot_patients_matrix(ids[i], spacings[i], **okwargs, **kwargs)

    # Add moved landmarks.
    if moved_landmark_data is not None and show_moved_landmarks:
        okwargs = dict(
            colour=moved_colour,
            zorder=0,
        )
        __plot_landmark_data(moved_landmark_data, axs[0], fontsize, idxs[0], sizes[0], view, **okwargs, **kwargs)

    if show_grid:
        # Plot moving grid.
        include = [True] * 3
        include[view] = False
        moving_grid = __create_grid(moving_ct_data.shape, moving_spacing, include=include)
        moving_idx = __convert_float_idx(moving_idx, moving_ct_data.shape, view)
        if show_moving:
            grid_slice = __get_mpl_slice(moving_grid, moving_idx, view)
            aspect = __get_mpl_aspect(view, moving_spacing)
            origin = __get_mpl_origin(view)
            axs[n_figs].imshow(grid_slice, aspect=aspect, cmap='gray', origin=origin)

        # Plot moved grid.
        moved_idx = __convert_float_idx(moved_idx, moved_ct_data.shape, view)
        if transform_format == 'itk':
            moved_grid = itk_transform_image(moving_grid, moving_spacing, moving_offset, fixed_ct_data.shape, fixed_spacing, fixed_offset, transform)
        elif transform_format == 'sitk':
            moved_grid = sitk_transform_image(moving_grid, moving_spacing, moving_offset, fixed_ct_data.shape, fixed_spacing, fixed_offset, transform)
        grid_slice = __get_mpl_slice(moved_grid, moving_idx, view)
        aspect = __get_mpl_aspect(view, fixed_spacing)
        origin = __get_mpl_origin(view)
        axs[2 * n_figs - 1].imshow(grid_slice, aspect=aspect, cmap='gray', origin=origin)

    if show_region_overlay:
        # Plot fixed/moved regions.
        aspect = __get_mpl_aspect(view, fixed_spacing)
        okwargs = dict(
            alpha=alpha_region,
            crop=fixed_crop,
            view=view,
        )
        f_idx = __convert_float_idx(fixed_idx, fixed_ct_data.shape, view)
        background = __get_mpl_slice(np.zeros(shape=fixed_ct_data.shape), f_idx, view)
        if fixed_crop is not None:
            background = crop(background, __reverse_box_coords_2D(fixed_crop))
        axs[2 * n_figs - 2].imshow(background, cmap='gray', aspect=aspect, interpolation='none', origin=__get_mpl_origin(view))
        if fixed_region_data is not None:
            __plot_region_data(fixed_region_data, axs[2 * n_figs - 2], f_idx, aspect, **okwargs)
        if moved_region_data is not None:
            __plot_region_data(moved_region_data, axs[2 * n_figs - 2], f_idx, aspect, **okwargs)

def apply_region_labels(
    region_labels: Dict[str, str],
    region_data: Optional[Dict[str, np.ndarray]],
    centre: Optional[str],
    crop: Optional[str]) -> Tuple[Dict[str, np.ndarray], Optional[str], Optional[str]]:

    # Apply region labels to 'region_data' and 'centre/crop' keys.
    if region_data is not None:
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)
    if centre is not None and type(centre) == str and centre in region_labels:
        centre = region_labels[centre] 
    if centre is not None and type(crop) == str and crop in region_labels:
        crop = region_labels[crop]

    return region_data, centre, crop

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

def __calculate_p_val(
    a: List[float],
    b: List[float],
    stats_alt: AltHyp,
    stats_paired: bool,
    stats_sig: List[float]) -> float:

    if stats_paired:
        if np.any(a.isna()) or np.any(b.isna()):
            raise ValueError(f"Unpaired data... add more info.")

        # Can't calculate paired test without differences.
        if np.all(b - a == 0):
            raise ValueError(f"Paired data is identical ... add more info.")
    else:
        # Remove 'nan' values.
        a = a[~a.isna()]
        b = b[~b.isna()]

    # Check for presence of data.
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"Empty data... add more info.")

    # Calculate p-value.
    alt_map = {
        AltHyp.LESSER: 'less',
        AltHyp.GREATER: 'greater',
        AltHyp.TWO_SIDED: 'two-sided'
    }
    if stats_paired:
        _, p_val = scipy.stats.wilcoxon(a, b, alternative=alt_map[stats_alt])
    else:
        _, p_val = scipy.stats.mannwhitneyu(a, b, alternative=alt_map[stats_alt])

    return p_val

def __format_p_vals(
    p_vals: List[float],
    stats_sig: List[float]) -> List[str]:
    p_vals_f = []
    for p in p_vals:
        p_f = ''
        for s in stats_sig:
            if p < s:
                p_f += '*'
            else:
                break
        p_vals_f.append(p_f)
    return p_vals_f

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

def __set_axes_limits(
    parent_ax: mpl.axes.Axes,
    x_lim: Tuple[float, float],
    y_lim: Tuple[float, float],
    inset_ax: Optional[mpl.axes.Axes] = None) -> mpl.axes.Axes:
    # Set parent axes limits.
    parent_ax.set_xlim(*x_lim)
    parent_ax.set_ylim(*y_lim)
    parent_ax.axis('off')

    # Create inset if not passed.
    if inset_ax is None:
        inset_ax = parent_ax.inset_axes([x_lim[0], y_lim[0], x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]], transform=parent_ax.transData)

    # Set inset axis limits.
    inset_ax.set_xlim(*x_lim)
    inset_ax.set_ylim(*y_lim)

    return inset_ax

def __view_to_text(
    view: int,
    abbreviate: bool = True) -> str:
    if view == 0:
        return 'sag.' if abbreviate else 'sagittal'
    elif view == 1:
        return 'cor.' if abbreviate else 'coronal'
    elif view == 2:
        return 'ax.' if abbreviate else 'axial'

def __convert_figsize_to_inches(figsize: Tuple[float, float]) -> Tuple[float, float]:
    cm_to_inch = 1 / 2.54
    figsize = figsize[0] * cm_to_inch if figsize[0] is not None else None, figsize[1] * cm_to_inch if figsize[1] is not None else None
    return figsize

def __create_grid(
    size: ImageSize3D,
    spacing: ImageSpacing3D,
    # We may not want lines running in all directions as it might look convoluted when warped.
    include: Tuple[bool] = (True, True, True),
    grid_intensity: int = 1,
    # Determines grid spacing in mm.
    grid_spacing: Tuple[int] = (25, 25, 25),
    linewidth: int = 1,
    use_shading: bool = False) -> np.ndarray:
    # Create grid.
    grid = np.zeros(size, dtype=np.float32)
    offset = [-1, -1, -1]
    grid_spacing_voxels = np.array(grid_spacing) / spacing

    for axis in range(3):
        if include[axis]:
            # Get line positions.
            line_idxs = [i for i in list(np.arange(grid.shape[axis])) if int(np.floor((i - offset[axis]) % grid_spacing_voxels[axis])) in tuple(range(linewidth))]

            # Set lines in grid image.
            idxs = [slice(None), slice(None), slice(None)]
            if use_shading:
                for i, idx in enumerate(line_idxs):
                    idxs[axis] = idx
                    grid[tuple(idxs)] = (i + 1) * grid_intensity
            else:
                idxs[axis] = line_idxs
                grid[tuple(idxs)] = grid_intensity
        
    return grid

def __convert_float_idx(
    idx: Optional[Union[int, float]],
    size: ImageSize3D,
    view: Axis) -> int:
    # Map float idx (\in [0, 1]) to int idx (\in [0, size[view]]).
    if idx is not None and idx > 0 and idx < 1:
        idx = int(np.floor(idx * size[view]))
    
    return idx

def __plot_landmark_data(
    landmark_data: Landmarks,
    ax: mpl.axes.Axes,
    fontsize: float,
    idx: Union[int, float],
    size: ImageSize3D,
    view: Axis, 
    colour: str = 'yellow',
    n_landmarks: Optional[int] = None,
    show_landmark_ids: bool = False,
    zorder: float = 1,
    **kwargs) -> None:
    idx = __convert_float_idx(idx, size, view)
    landmark_data = landmark_data.copy()
    # Take subset of n closest landmarks landmarks.
    landmark_data['dist'] = np.abs(landmark_data[view] - idx)
    if n_landmarks is not None:
        landmark_data = landmark_data.sort_values('dist')
        landmark_data = landmark_data.iloc[:n_landmarks]

    # Convert distance to alpha.
    dist_norm = landmark_data['dist'] / (size[view] / 2)
    dist_norm[dist_norm > 1] = 1
    landmark_data['dist-norm'] = dist_norm
    base_alpha = 0.3
    landmark_data['alpha'] = base_alpha + (1 - base_alpha) * (1 - landmark_data['dist-norm'])
    
    r, g, b = to_rgb(colour)
    colours = [(r, g, b, a) for a in landmark_data['alpha']]
    img_axs = list(range(3))
    img_axs.remove(view)
    lm_x, lm_y, lm_ids = landmark_data[img_axs[0]], landmark_data[img_axs[1]], landmark_data['landmark-id']
    ax.scatter(lm_x, lm_y, c=colours, s=20, zorder=zorder)
    if show_landmark_ids:
        for x, y, t in zip(lm_x, lm_y, lm_ids):
            ax.text(x, y, t, fontsize=fontsize, color='white')

def sanitise_label(
    s: str,
    max_length: int = 25) -> str:
    s = s.lstrip('_')
    if len(s) > max_length:
        s = s[:max_length]
    return s

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pdf2image import convert_from_path
import seaborn as sns
from tqdm import tqdm
from typing import *

from mymi.datasets import Dataset
from mymi.geometry import get_box, get_centre_of_mass, fov_centre
from mymi import logging
from mymi.processing import largest_cc_3D
from mymi.regions import get_region_patch_size
from mymi.regions import truncate_spine as truncate
from mymi.transforms import crop_or_pad_box, crop_point, crop as crop_fn, itk_transform_image, replace_box_none, resample, sample
from mymi.typing import *
from mymi.utils import *

def box_intersects_view_plane(
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

def escape_latex_fn(text: str) -> str:
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

def figsize_to_inches(figsize: Tuple[float, float]) -> Tuple[float, float]:
    cm_to_inch = 1 / 2.54
    figsize = figsize[0] * cm_to_inch if figsize[0] is not None else None, figsize[1] * cm_to_inch if figsize[1] is not None else None
    return figsize

def get_aspect(
    view: Axis,
    spacing: Spacing3D) -> float:
    if view == 0:
        aspect = spacing[2] / spacing[1]
    elif view == 1:
        aspect = spacing[2] / spacing[0]
    elif view == 2:
        aspect = spacing[1] / spacing[0]
    return np.abs(aspect)

def get_idx(
    size: Size3D,
    view: Axis,
    centre: Optional[Union[LandmarkData, LandmarkID, Literal['dose'], RegionArray, RegionID]] = None,
    centre_other: bool = False,
    dose_data: Optional[DoseData] = None,
    idx: Optional[Union[int, float]] = None,
    idx_mm: Optional[float] = None,
    landmark_data: Optional[LandmarksData] = None,
    landmark_data_other: Optional[LandmarksData] = None,
    offset: Optional[Point3D] = None,
    region_data: Optional[RegionsData] = None,
    spacing: Optional[Spacing3D] = None) -> int:
    if idx is not None:
        if centre is not None:
            if isinstance(centre, (LandmarkData, RegionArray)):
                centre = type(centre)
            raise ValueError(f"Cannot specify both 'centre' ({centre}) and 'idx' ({idx}).")
        if idx_mm is not None:
            raise ValueError(f"Cannot specify both 'idx' and 'idx_mm'.")
        if idx > 0 and idx < 1:
            # Map float idx (\in [0, 1]) to int idx (\in [0, size[view]]).
            idx = int(np.round(idx * (size[view] - 1)))
    elif idx_mm is not None:
        if centre is not None:
            raise ValueError(f"Cannot specify both 'centre' and 'idx_mm'.")
        # Find nearest voxel index to mm position.
        idx = int(np.round((idx_mm - offset[view]) / spacing[view]))
    elif centre is not None:
        if isinstance(centre, str) and centre == 'dose':
            if dose_data is None:
                raise ValueError("Cannot use 'dose' centre without 'dose_data'.")
            centre = get_centre_of_mass(dose_data, use_patient_coords=False)
            idx = centre[view]
        elif isinstance(centre, (LandmarkID, RegionID)):
            lm_data = landmark_data_other if centre_other and landmark_data_other is not None else landmark_data
            if lm_data is not None and centre in list(lm_data['landmark-id']):
                centre_point = lm_data[lm_data['landmark-id'] == centre][list(range(3))].iloc[0]
                idx = point_to_image_coords(centre_point, spacing, offset)[view]
            elif region_data is not None and centre in region_data:
                idx = fov_centre(region_data[centre], use_patient_coords=False)[view]
            else:
                raise ValueError(f"No centre '{centre}' found in 'landmarks/region_data'.")
        elif isinstance(centre, LandmarkData):
            centre_point = tuple(centre[list(range(3))])
            idx = point_to_image_coords(centre_point, spacing, offset)[view]
        elif isinstance(centre, RegionArray):
            idx = fov_centre(centre, use_patient_coords=False)[view]
        else:
            raise ValueError(f"Invalid type for 'centre': {type(centre)}. Must be one of (LandmarkData, LandmarkID, RegionArray, RegionID).")
    else:
        # raise ValueError(f"Either 'centre', 'idx' or 'idx_mm' must be specified.")
        idx = np.round(size[view] / 2).astype(int)

    return idx

def get_origin(view: Axis) -> Literal['lower', 'upper']:
    return 'upper' if view == 2 else 'lower'

def get_view_slice(
    data: Union[ImageArray3D, VectorImageArray],
    idx: Union[int, float],
    view: Axis) -> Tuple[ImageArray2D, int]:
    n_dims = len(data.shape)
    if n_dims == 4:
        assert data.shape[0] == 3   # vector image.
        view_idx = view + 1
    else:
        view_idx = view

    # Check that slice index isn't too large.
    if idx >= data.shape[view_idx]:
        raise ValueError(f"Idx '{idx}' out of bounds, only '{data.shape[view_idx]}' {get_axis_name(view)} indices.")
    
    # Handle index in range [0, 1).
    if isinstance(idx, float) and idx >= 0 and idx < 1:
        idx = int(np.round(idx * data.shape[view_idx]))

    # Get correct plane.
    data_index = [slice(None)] if n_dims == 4 else []
    for i in range(3):
        v_idx = i + 1 if n_dims == 4 else i
        data_index += [idx if i == view else slice(data.shape[v_idx])]
    data_index = tuple(data_index)
    slice_data = data[data_index]

    # Convert to 'imshow' coords.
    slice_data = np.transpose(slice_data)

    return slice_data, idx

def get_view_xy(
    data: Union[Spacing3D, Point3D],
    view: Axis) -> Tuple[float, float]:
    if view == 0:
        res = (data[1], data[2])
    elif view == 1:
        res = (data[0], data[2])
    elif view == 2:
        res = (data[0], data[1])
    return res

def get_window(
    window: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    data: Optional[ImageArray3D] = None) -> Tuple[float, float]:
    if isinstance(window, tuple):
        width, level = window
    elif isinstance(window, str):
        if window == 'bone':
            width, level = (1800, 400)
        elif window == 'lung':
            width, level = (1500, -600)
        elif window == 'tissue':
            width, level = (400, 40)
        else:
            raise ValueError(f"Window '{window}' not recognised.")
    elif window is None:
        if data is not None:
            width, level = data.max() - data.min(), (data.min() + data.max()) / 2
        else:
            width, level = (0, 0)
    else:
        raise ValueError(f"Unrecognised type for 'window' ({type(window)}).")

    if data is not None:
        # Check that CT data isn't going to be hidden.
        data_min, data_max = data.min(), data.max()
        data_width = data_max - data_min
        f = 0.1
        if data_width < f * width:
            logging.warning(f"Image data range ({data_min}, {data_max}) is less than {f} * window range ({width}). You may be looking at grey - use a custom window (level, width).")

    vmin = level - (width / 2)
    vmax = level + (width / 2)

    return vmin, vmax

def plot_box_slice(
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
    min, max = box
    min = __get_view_xy(min, view)
    max = __get_view_xy(max, view)
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

def plot_landmark_data(
    landmark_data: Union[LandmarksData, LandmarksDataVox, Points3D],
    ax: mpl.axes.Axes,
    idx: int,
    size: Size3D,
    spacing: Spacing3D,
    offset: Point3D,
    view: Axis, 
    colour: str = 'yellow',
    crop: Optional[Box2D] = None,
    dose_data: Optional[DoseData] = None,
    fontsize_landmarks: float = 10,
    landmark_ids: LandmarkIDs = 'all',
    landmarks_use_patient_coords: bool = True,
    n_landmarks: Optional[int] = None,
    show_landmark_ids: bool = False,
    show_landmark_dists: bool = True,
    show_landmark_doses: bool = False,
    zorder: float = 1,
    **kwargs) -> None:
    landmark_data = landmark_data.copy()
    if isinstance(landmark_data, Points3D):
        landmark_data = landmarks_from_data(landmark_data)

    # Filter by 'landmark_ids'.
    if landmark_ids != 'all':
        landmark_ids = arg_to_list(landmark_ids, (int, str))
        landmark_data = landmark_data[landmark_data['landmark-id'].isin(landmark_ids)]

    # Add sampled dose intensities.
    if show_landmark_doses and dose_data is not None:
        landmark_data = sample(dose_data, landmark_data, landmarks_col='dose', offset=offset, spacing=spacing)

    # Convert landmarks to image coords.
    if landmarks_use_patient_coords:
        landmark_data = landmarks_to_image_coords(landmark_data, spacing, offset)

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
    
    r, g, b = mpl.colors.to_rgb(colour)
    if show_landmark_dists:
        colours = [(r, g, b, a) for a in landmark_data['alpha']]
    else:
        colours = [(r, g, b, 1) for _ in landmark_data['alpha']]
    img_axs = list(range(3))
    img_axs.remove(view)
    lm_x, lm_y, lm_ids = landmark_data[img_axs[0]], landmark_data[img_axs[1]], landmark_data['landmark-id']
    lm_doses = landmark_data['dose'].values if 'dose' in landmark_data.columns else None
    if crop is not None:
        lm_x, lm_y = lm_x - crop[0][0], lm_y - crop[0][1]
    ax.scatter(lm_x, lm_y, c=colours, s=20, zorder=zorder)
    if show_landmark_doses or show_landmark_ids:
        for i, (id, x, y) in enumerate(zip(lm_ids, lm_x, lm_y)):
            if show_landmark_ids:
                text = f'{id} ({lm_doses[i]:.1f})' if lm_doses is not None else id
            else:
                text = f'{lm_doses[i]:.1f}'
            ax.text(x, y, text, fontsize=fontsize_landmarks, color=colour)

def plot_region_data(
    data: RegionsData,
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
        cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), colour))

        # Convert data to 'imshow' co-ordinate system.
        slice_data, _ = get_view_slice(data[region], idx, view)

        # Crop image.
        if crop:
            slice_data = crop_fn(slice_data, transpose_box(crop), use_patient_coords=False)

        # Plot extent.
        if show_extent:
            ext = extent(data[region])
            if ext is not None:
                label = f'{region} extent' if box_intesects_view_plane(ext, view, idx) else f'{region} extent (offscreen)'
                plot_box_slice(ext, view, ax=ax, colour=colour, crop=crop, label=label, linestyle='dashed')
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
        ax.imshow(slice_data, alpha=alpha, aspect=aspect, cmap=cmap, interpolation='none', origin=get_origin(view))
        label = escape_latex_fn(region) if escape_latex else region
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

def plot_saved(
    filepaths: Union[str, List[str]],
    figwidth: float = 16) -> None:
    filepaths = arg_to_list(filepaths, str)
    for f in filepaths:
        f = escape_filepath(f)
        # images = convert_from_path(f)
        # image = images[0]
        image = plt.imread(f)
        plt.figure(figsize=(figwidth, figwidth * image.shape[0] / image.shape[1]))
        plt.axis('off')
        plt.imshow(image)

def transpose_box(box: Union[Box2D, Box3D]) -> Union[Box2D, Box3D]:
    return tuple(tuple(reversed(b)) for b in box)
    
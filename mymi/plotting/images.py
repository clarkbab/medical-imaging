import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *

from mymi.typing import *
from mymi.utils import *

from .plotting import get_aspect, get_origin, get_view_slice, get_view_xy, get_window, plot_landmark_data

def plot_images(
    data: Union[ImageData3D, List[ImageData3D]],
    figsize: Tuple[float, float] = (16, 6),
    idxs: Union[int, float, List[Union[int, float]]] = 0.5,
    labels: Optional[Union[LabelData3D, List[Optional[LabelData3D]]]] = None,
    landmarks: Optional[Union[LandmarksData, List[LandmarksData]]] = None,    # Should be in patient coordinates.
    landmark_ids: LandmarkIDs = 'all',
    modality: Literal['ct', 'dose'] = 'ct',
    offsets: Optional[Union[Point3D, List[Point3D]]] = (0, 0, 0),
    points: Optional[Union[Point3D, List[Point3D]]] = None,
    spacings: Optional[Union[Spacing3D, List[Spacing3D]]] = (1, 1, 1),
    transpose: bool = False,
    use_patient_coords: bool = False,
    views: Union[int, Sequence[int]] = 'all',
    window: Optional[Union[str, Tuple[float, float]]] = None,
    **kwargs) -> None:
    data = arg_to_list(data, ImageData3D)
    idxs = arg_to_list(idxs, (int, float), broadcast=len(data))
    labels = arg_to_list(labels, [LabelData3D, None], broadcast=len(data))
    landmarks = arg_to_list(landmarks, [LandmarksData, None], broadcast=len(data))
    offsets = arg_to_list(offsets, Point3D, broadcast=len(data))
    points = arg_to_list(points, [Point3D, None], broadcast=len(data))
    spacings = arg_to_list(spacings, Spacing3D, broadcast=len(data))
    assert len(labels) == len(data)
    assert len(landmarks) == len(data)
    assert len(offsets) == len(data)
    assert len(spacings) == len(data)
    palette = sns.color_palette('colorblind', len(labels))
    views = arg_to_list(views, int, literals={ 'all': list(range(3)) })
    n_rows, n_cols = (len(views), len(data)) if transpose else (len(data), len(views))
    _, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0], len(data) * figsize[1]), squeeze=False)
    for i, (row_axs, d, idx, l, lm, o, s, p) in enumerate(zip(axs, data, idxs, labels, landmarks, offsets, spacings, points)):
        # Rescale RGB image to range [0, 1).
        n_dims = len(d.shape)
        if n_dims == 4:
            d = (d - d.min()) / (d.max() - d.min())

        for col_ax, v in zip(row_axs, views):
            image, view_idx = get_view_slice(d, idx, v)
            aspect = get_aspect(v, s)
            origin = get_origin(v)
            vmin, vmax = get_window(window, d)
            if modality == 'ct':
                cmap='gray'
            elif modality == 'dose':
                cmap='viridis'
            col_ax.imshow(image, aspect=aspect, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
            col_ax.set_title(f'{get_axis_name(v)} view, slice {view_idx}')
            if l is not None:   # Plot labels.
                cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[i]))
                label_image, _ = get_view_slice(l, idx, v)
                col_ax.imshow(label_image, alpha=0.3, aspect=aspect, cmap=cmap, origin=origin)
                col_ax.contour(label_image, colors=[palette[i]], levels=[.5], linestyles='solid')
            if use_patient_coords:  # Change axis tick labels to show patient coordinates.
                size_x, size_y = get_view_xy(d.shape, v)
                sx, sy = get_view_xy(s, v)
                ox, oy = get_view_xy(o, v)
                x_tick_spacing = np.unique(np.diff(col_ax.get_xticks()))[0]
                x_ticks = np.arange(0, size_x, x_tick_spacing)
                x_ticklabels = x_ticks * sx + ox
                x_ticklabels = [f'{l:.1f}' for l in x_ticklabels]
                col_ax.set_xticks(x_ticks)
                col_ax.set_xticklabels(x_ticklabels)
                y_tick_spacing = np.unique(np.diff(col_ax.get_yticks()))[0]
                y_ticks = np.arange(0, size_y, y_tick_spacing)
                y_ticklabels = y_ticks * sy + oy
                y_ticklabels = [f'{l:.1f}' for l in y_ticklabels]
                col_ax.set_yticks(y_ticks)
                col_ax.set_yticklabels(y_ticklabels)
                view_loc = view_idx * s[v] + o[v]
                col_ax.set_title(f'{get_axis_name(v)} view, slice {view_idx} ({view_loc:.1f}mm)')
            if lm is not None:
                plot_landmark_data(lm, col_ax, view_idx, d.shape, s, o, v, landmark_ids=landmark_ids)
            if p is not None:
                # Convert point to image coordinates.
                pv = (np.array(p) - o) / s
                px, py = get_view_xy(pv, v)
                col_ax.scatter(px, py, c='yellow', s=20, zorder=1)

@delegates(plot_images)
def plot_nifti(
    filepath: str,
    spacing: Optional[Spacing3D] = None,
    offset: Optional[Point3D] = None,
    **kwargs) -> None:
    data, nspacing, noffset = load_nifti(filepath)
    spacing = nspacing if spacing is None else spacing
    offset = noffset if offset is None else offset
    plot_images(data, offsets=offset, spacings=spacing, **kwargs)

@delegates(load_numpy, plot_images)
def plot_numpy(
    filepath: str,
    spacing: Optional[Spacing3D] = (1, 1, 1),
    offset: Optional[Point3D] = (0, 0, 0),
    **kwargs) -> None:
    data = load_numpy(filepath, **kwargs)
    plot_images(data, offsets=offset, spacings=spacing, **kwargs)

def sitk_plot_image(
    filepath: str,
    spacing: Optional[Spacing3D] = None,
    offset: Optional[Point3D] = None,
    **kwargs) -> None:
    data, lspacing, loffset = sitk_load_image(filepath)
    if spacing is None:
        spacing = lspacing
    if offset is None:
        offset = loffset
    plot_images(data, offsets=offset, spacings=spacing, **kwargs)

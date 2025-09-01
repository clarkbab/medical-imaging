import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *

from mymi.typing import *
from mymi.utils import *

from .plotting import get_aspect, get_origin, get_view_slice, get_view_xy, get_window, plot_landmark_data

@alias_kwargs(('upc', 'use_patient_coords'))
def plot_image(
    data: Union[ImageArray, ImageTensor, DirPath, FilePath, List[Union[ImageArray, ImageTensor]]],
    figsize: Tuple[float, float] = (16, 6),
    idx: Union[int, float, List[Union[int, float]]] = 0.5,
    # If single or list, broadcast to all images. If list of lists, leave alone.
    label: Optional[Union[LabelArray, LabelTensor, DirPath, FilePath, List[Union[LabelArray, LabelTensor]], List[List[Union[LabelArray, LabelTensor]]]]] = None,
    landmarks: Optional[Union[LandmarksFrame, PointsArray, PointsTensor, List[Union[LandmarksFrame, PointsArray, PointsTensor]]]] = None,    # Should be in patient coordinates.
    landmark_ids: LandmarkIDs = 'all',
    modality: Literal['ct', 'dose'] = 'ct',
    origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = (0, 0, 0),
    spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = (1, 1, 1),
    transpose: bool = False,
    use_patient_coords: bool = False,
    view: Union[int, Sequence[int]] = 'all',
    window: Optional[Union[str, Tuple[float, float]]] = None,
    **kwargs) -> None:
    if isinstance(data, (DirPath, FilePath)):
        if os.path.isdir(data):
            data, spacings, origins = from_ct_dicoms(dirpath=data)
        elif data.endswith('.nii') or data.endswith('.nii.gz'):
            data, spacings, origins = load_nifti(data)
        elif data.endswith('.npy'):
            data, spacings, origins = load_numpy(data)
        elif data.endswith('.nrrd'):
            data, spacings, origins = load_nrrd(data)
        elif data.endswith('.mha') or data.endswith('.mhd'):
            data, spacings, origins = sitk_load_image(data)
        else:
            raise ValueError(f'Unsupported file type: {data}')
    data = arg_to_list(data, (np.ndarray, torch.Tensor))
    idxs = arg_to_list(idx, (int, float), broadcast=len(data))
    # Assuming one main image only.
    if isinstance(label, (DirPath, FilePath)):
        loaded_labels = []
        if os.path.isdir(label):
            for f in os.listdir(label):
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    l, _, _ = load_nifti(os.path.join(label, f))
                loaded_labels.append(l)
        elif label.endswith('.nii') or label.endswith('.nii.gz'):
            loaded_label, _, _ = load_nifti(label)
            loaded_labels.append(loaded_label)
        labels = [loaded_labels]
    labels = arg_to_list(label, (None, np.ndarray, torch.Tensor))   # From single element to list.
    labels = arg_to_list(labels, list, broadcast=len(data))   # From list to list of lists.
    landmarks = arg_to_list(landmarks, (None, pd.DataFrame, np.ndarray, torch.Tensor), broadcast=len(data))
    spacings = arg_to_list(spacing, (tuple, np.ndarray, torch.Tensor), broadcast=len(data))
    origins = arg_to_list(origin, (tuple, np.ndarray, torch.Tensor), broadcast=len(data))
    assert len(landmarks) == len(data)
    assert len(origins) == len(data)
    assert len(spacings) == len(data)
    n_label_max = np.max([len(ls) if ls is not None else 0 for ls in labels])
    palette = sns.color_palette('colorblind', n_label_max)
    views = arg_to_list(view, int, literals={ 'all': list(range(3)) })
    n_rows, n_cols = (len(views), len(data)) if transpose else (len(data), len(views))

    # Convert tensors.
    for i in range(len(data)):
        if isinstance(data[i], torch.Tensor):
            data[i] = data[i].cpu().numpy()
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if isinstance(labels[i][j], torch.Tensor):
                labels[i][j] = labels[i][j].cpu().numpy()
    for i in range(len(landmarks)):
        if isinstance(landmarks[i], torch.Tensor):
            landmarks[i] = landmarks[i].cpu().numpy()
    for i in range(len(spacings)):
        if isinstance(spacings[i], torch.Tensor):
            spacings[i] = spacings[i].cpu().numpy()
    for i in range(len(origins)):
        if isinstance(origins[i], torch.Tensor):
            origins[i] = origins[i].cpu().numpy()

    # Plot images.
    _, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0], len(data) * figsize[1]), squeeze=False)
    for i, (row_axs, d, idx, ls, lm, o, s) in enumerate(zip(axs, data, idxs, labels, landmarks, origins, spacings)):
        logging.info(f"Plotting image {i+1}/{len(data)}: with size={d.shape}, idx={idx}, spacing={s}, origin={o}.")

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

            # Plot labels.
            for j, l in enumerate(ls):
                if l is None:
                    continue
                cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[j]))
                assert len(l.shape) == 3
                label_image, _ = get_view_slice(l, idx, v)
                col_ax.imshow(label_image, alpha=0.3, aspect=aspect, cmap=cmap, origin=origin)
                col_ax.contour(label_image, colors=[palette[j]], levels=[.5], linestyles='solid')

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

@delegates(plot_image)
def plot_nifti(
    filepath: str,
    spacing: Optional[Spacing3D] = None,
    origin: Optional[Point3D] = None,
    **kwargs) -> None:
    data, nspacing, norigin = load_nifti(filepath)
    spacing = nspacing if spacing is None else spacing
    origin = norigin if origin is None else origin
    plot_image(data, origin=origin, spacing=spacing, **kwargs)

@delegates(load_numpy, plot_image)
def plot_numpy(
    filepath: str,
    spacing: Optional[Spacing3D] = (1, 1, 1),
    origin: Optional[Point3D] = (0, 0, 0),
    **kwargs) -> None:
    data = load_numpy(filepath, **kwargs)
    plot_image(data, origin=origin, spacing=spacing, **kwargs)

def sitk_plot_image(
    filepath: str,
    spacing: Optional[Spacing3D] = None,
    origin: Optional[Point3D] = None,
    **kwargs) -> None:
    data, lspacing, lorigin = sitk_load_image(filepath)
    if spacing is None:
        spacing = lspacing
    if origin is None:
        origin = lorigin
    plot_image(data, origin=origin, spacing=spacing, **kwargs)

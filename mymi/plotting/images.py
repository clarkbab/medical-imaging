import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *

from mymi.typing import *
from mymi.utils import *

from .plotting import get_aspect, get_idx, get_origin, get_view_slice, get_view_xy, get_window, plot_landmarks_data

@alias_kwargs(('upc', 'use_patient_coords'))
def plot_image(
    data: Union[ImageArray, ImageTensor, DirPath, FilePath, List[Union[ImageArray, ImageTensor]]],
    centre: Optional[Union[LandmarkID, LandmarkSeries, Literal['dose'], RegionArray, RegionID]] = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    dose_data: Union[ImageArray, ImageTensor, DirPath, FilePath, List[Union[ImageArray, ImageTensor]]] = None,
    dose_origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = None,
    dose_spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = None,
    ocentre: Optional[Union[LandmarkID, LandmarkSeries, Literal['dose'], RegionArray, RegionID]] = None,
    figsize: Tuple[float, float] = (16, 6),
    idx: Union[int, float, List[Union[int, float]]] = 0.5,
    # If single or list, broadcast to all images. If list of lists, leave alone.
    region: Optional[Union[LabelArray, LabelTensor, DirPath, FilePath, List[Union[LabelArray, LabelTensor]], List[List[Union[LabelArray, LabelTensor]]]]] = None,
    landmark: LandmarkIDs = 'all',
    landmarks_data: Optional[Union[LandmarksFrame, PointsArray, PointsTensor, List[Union[LandmarksFrame, PointsArray, PointsTensor]]]] = None,    # Should be in patient coordinates.
    modality: Literal['ct', 'dose'] = 'ct',
    origin: Optional[Union[Point, PointArray, PointTensor, List[Union[Point, PointArray, PointTensor]]]] = (0, 0, 0),
    other_landmarks_data: Optional[Union[LandmarksFrame, PointsArray, PointsTensor, List[Union[LandmarksFrame, PointsArray, PointsTensor]]]] = None,    # Should be in patient coordinates.
    show_axis_ticks: bool = True,
    show_axis_tick_labels: bool = True,
    show_dose: bool = True,
    show_landmarks: bool = True,    # Can pass 'landmarks_data' for 'centre' but don't want plotted.
    show_title: bool = True,
    spacing: Optional[Union[Spacing, SpacingArray, SpacingTensor, List[Union[Spacing, SpacingArray, SpacingTensor]]]] = (1, 1, 1),
    transpose: bool = False,
    use_patient_coords: bool = True,
    view: Union[int, List[int]] = 'all',
    window: Optional[Union[str, Tuple[float, float]]] = 'tissue',
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
    dose_data = arg_to_list(dose_data, (None, np.ndarray, torch.Tensor))
    idxs = arg_to_list(idx, (int, float, str), broadcast=len(data))
    centres = arg_to_list(centre, (None, int, float, str), broadcast=len(data))
    ocentres = arg_to_list(ocentre, (None, int, float, str), broadcast=len(data))
    # Assuming one main image only.
    if isinstance(region, (DirPath, FilePath)):
        loaded_regions = []
        if os.path.isdir(region):
            for f in os.listdir(region):
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    l, _, _ = load_nifti(os.path.join(region, f))
                loaded_regions.append(l)
        elif region.endswith('.nii') or region.endswith('.nii.gz'):
            loaded_region, _, _ = load_nifti(region)
            loaded_regions.append(loaded_region)
        regions = [loaded_regions]
    regions = arg_to_list(region, (None, np.ndarray, torch.Tensor))   # From single element to list.
    regions = arg_to_list(regions, list, broadcast=len(data))   # From list to list of lists.
    landmarks_datas = arg_to_list(landmarks_data, (None, pd.DataFrame, np.ndarray, torch.Tensor), broadcast=len(data))
    show_doses = arg_to_list(show_dose, (bool), broadcast=len(data))
    show_landmarkses = arg_to_list(show_landmarks, (bool), broadcast=len(data))
    other_landmarks_datas = arg_to_list(other_landmarks_data, (None, pd.DataFrame, np.ndarray, torch.Tensor), broadcast=len(data))
    spacings = arg_to_list(spacing, (tuple, np.ndarray, torch.Tensor), broadcast=len(data))
    dose_spacings = arg_to_list(dose_spacing, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(data))
    origins = arg_to_list(origin, (tuple, np.ndarray, torch.Tensor), broadcast=len(data))
    dose_origins = arg_to_list(dose_origin, (None, tuple, np.ndarray, torch.Tensor), broadcast=len(data))
    assert len(landmarks_datas) == len(data)
    assert len(other_landmarks_datas) == len(data)
    assert len(origins) == len(data)
    assert len(spacings) == len(data)
    n_region_max = np.max([len(ls) if ls is not None else 0 for ls in regions])
    palette = sns.color_palette('colorblind', n_region_max)
    views = arg_to_list(view, int, literals={ 'all': list(range(3)) })

    # Convert tensors.
    for i in range(len(data)):
        if isinstance(data[i], torch.Tensor):
            data[i] = data[i].cpu().numpy()
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            if isinstance(regions[i][j], torch.Tensor):
                regions[i][j] = regions[i][j].cpu().numpy()
    for i in range(len(landmarks_datas)):
        if isinstance(landmarks_datas[i], torch.Tensor):
            landmarks_datas[i] = landmarks_datas[i].cpu().numpy()
    for i in range(len(other_landmarks_datas)):
        if isinstance(other_landmarks_datas[i], torch.Tensor):
            other_landmarks_datas[i] = other_landmarks_datas[i].cpu().numpy()
    for i in range(len(spacings)):
        if isinstance(spacings[i], torch.Tensor):
            spacings[i] = spacings[i].cpu().numpy()
    for i in range(len(origins)):
        if isinstance(origins[i], torch.Tensor):
            origins[i] = origins[i].cpu().numpy()

    # Plot images.
    n_rows, n_cols = (len(views), len(data)) if transpose else (len(data), len(views))
    _, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0], len(data) * figsize[1]), squeeze=False)
    if transpose:
        axs = axs.T

    for i, (row_axs, d, dd, idx, c, oc, rs, lms, o, do, olms, s, ds, sd, sl) in enumerate(zip(axs, data, dose_data, idxs, centres, ocentres, regions, landmarks_datas, origins, dose_origins, other_landmarks_datas, spacings, dose_spacings, show_doses, show_landmarkses)):
        logging.info(f"Plotting image {i+1}/{len(data)}: with size={d.shape}, idx={idx}, centre={c}, spacing={s}, origin={o}.")

        # Rescale RGB image to range [0, 1).
        n_dims = len(d.shape)
        if n_dims == 4:
            d = (d - d.min()) / (d.max() - d.min())

        for col_ax, v in zip(row_axs, views):
            view_idx = get_idx(d.shape, v, centre=c, ocentre=oc, idx=idx, landmarks_data=lms, landmarks_data_other=olms, origin=o, regions_data=rs, spacing=s)
            image, view_idx = get_view_slice(d, view_idx, v)
            aspect = get_aspect(v, s)
            origin = get_origin(v)
            vmin, vmax = get_window(window, d)
            if modality == 'ct':
                cmap='gray'
            elif modality == 'dose':
                cmap='viridis'
            col_ax.imshow(image, aspect=aspect, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
            if show_title:
                col_ax.set_title(f'{get_axis_name(v)} view, slice {view_idx}')
            if not show_axis_ticks:
                col_ax.set_xticks([])
                col_ax.set_yticks([])
            if not show_axis_tick_labels:
                col_ax.set_xticklabels([])
                col_ax.set_yticklabels([])

            # Plot dose.
            if sd and dd is not None:
                # Resample dose to image grid if necessary.
                if (ds is not None and ds != s) or (do is not None and do != o):
                    rs_params = dict(
                        origin=do if do is not None else o,
                        output_origin=o,
                        output_size=d.shape,
                        output_spacing=s,
                        spacing=ds if ds is not None else s,
                    )
                    dd = resample(dd, **rs_params)

                dose_image, view_idx = get_view_slice(dd, view_idx, v)

                # Create colormap with varying alpha - so 0 Gray is transparent.
                dose_cmap = plt.get_cmap(dose_cmap)

                # Remove purple from the colourmap.
                dose_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    f'{dose_cmap.name}_truncated',
                    dose_cmap(np.linspace(dose_cmap_trunc, 1.0, 256))
                )

                dose_colours = dose_cmap(np.arange(dose_cmap.N))
                dose_colours[0, -1] = 0
                dose_colours[1:, -1] = np.linspace(dose_alpha_min, dose_alpha_max, dose_cmap.N - 1)
                dose_cmap = mpl.colors.ListedColormap(dose_colours)

                # Plot dose slice.
                if show_dose:
                    imax = col_ax.imshow(dose_image, aspect=aspect, cmap=dose_cmap, origin=origin)
                    # if show_dose_legend:
                    #     cbar = plt.colorbar(imax, fraction=dose_colourbar_size, pad=dose_colourbar_pad)
                    #     cbar.set_label(label='Dose [Gray]', size=fontsize)
                    #     cbar.ax.tick_params(labelsize=fontsize)

            # Plot regions.
            for j, r in enumerate(rs):
                if r is None:
                    continue
                cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), palette[j]))
                assert len(r.shape) == 3
                region_image, _ = get_view_slice(r, idx, v)
                col_ax.imshow(region_image, alpha=0.3, aspect=aspect, cmap=cmap, origin=origin)
                col_ax.contour(region_image, colors=[palette[j]], levels=[.5], linestyles='solid')

            if use_patient_coords:  # Change axis tick labels to show patient coordinates.
                size_x, size_y = get_view_xy(d.shape, v)
                sx, sy = get_view_xy(s, v)
                ox, oy = get_view_xy(o, v)

                if show_axis_ticks:
                    x_tick_spacing = np.unique(np.diff(col_ax.get_xticks()))[0]
                    x_ticks = np.arange(0, size_x, x_tick_spacing)
                    col_ax.set_xticks(x_ticks)
                    y_tick_spacing = np.unique(np.diff(col_ax.get_yticks()))[0]
                    y_ticks = np.arange(0, size_y, y_tick_spacing)
                    col_ax.set_yticks(y_ticks)

                    if show_axis_tick_labels:
                        x_ticklabels = x_ticks * sx + ox
                        x_ticklabels = [f'{l:.1f}' for l in x_ticklabels]
                        col_ax.set_xticklabels(x_ticklabels)
                        y_ticklabels = y_ticks * sy + oy
                        y_ticklabels = [f'{l:.1f}' for l in y_ticklabels]
                        col_ax.set_yticks(y_ticks if show_axis_ticks else [])
                        col_ax.set_yticklabels(y_ticklabels)

                view_loc = view_idx * s[v] + o[v]
                if show_title:
                    col_ax.set_title(f'{get_axis_name(v)} view, slice {view_idx} ({view_loc:.1f}mm)')
            if sl and lms is not None:
                plot_landmarks_data(lms, col_ax, view_idx, d.shape, s, o, v, landmark=landmark, **kwargs)
            if sl and olms is not None:
                plot_landmarks_data(olms, col_ax, view_idx, d.shape, s, o, v, landmark=landmark, marker_colour='red', **kwargs)

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

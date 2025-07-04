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
from pdf2image import convert_from_path
import scipy
import seaborn as sns
from seaborn.palettes import _ColorPalette
import SimpleITK as sitk
import torchio
from tqdm import tqdm
from typing import *

from mymi.datasets import Dataset
from mymi.geometry import get_box, extent, centre_of_extent
from mymi import logging
from mymi.processing import largest_cc_3D
from mymi.regions import get_region_patch_size
from mymi.regions import truncate_spine as truncate
from mymi.transforms import crop_or_pad_box, crop_point, crop as crop_fn, itk_transform_image, replace_box_none, resample
from mymi.typing import *
from mymi.utils import *

DEFAULT_FONT_SIZE = 16

class AltHyp(Enum):
    LESSER = 0
    GREATER = 1
    TWO_SIDED = 2

def plot_histogram(
    data: Union[ImageData3D, str],
    ax: Optional[mpl.axes.Axes] = None,
    diff: Optional[ImageData3D] = None,
    fontsize: float = DEFAULT_FONT_SIZE,
    n_bins: int = 100,
    title: Optional[str] = None,
    x_lim: Optional[Tuple[Optional[float], Optional[float]]] = None) -> None:
    # Handle arguments.
    if isinstance(data, str):
        if data.endswith('.nii.gz'):
            data, _, _ = load_nifti(data)
    if ax is None:
        ax = plt.gca()
        show = True
    else:
        show = False
    
    # Plot histogram.
    if diff is not None:
        min_val = np.min([data.min(), diff.min()])
        max_val = np.max([data.max(), diff.max()])
    else:
        min_val, max_val = data.min(), data.max()
    bins = np.linspace(min_val, max_val, n_bins)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(data, bins=bins)
    if diff is not None:
        diff_hist, _ = np.histogram(diff, bins=bins)
        hist = hist - diff_hist
    colours = ['blue' if h >= 0 else 'red' for h in hist]
    ax.bar(bin_centres, hist, width=np.diff(bins), color=colours, edgecolor='black')
    ax.set_xlabel('Value', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # Add stats box.
    # Calculate stats on binned values to allow for diff calc.
    if diff is None:
        mean, std, median = data.mean(), data.std(), np.median(data)
    else:
        mean = np.sum(np.abs(hist) * bin_centres) / np.abs(hist).sum()
        std = np.sqrt(np.sum(np.abs(hist) * (bin_centres - mean) ** 2) / np.sum(np.abs(hist)))
        # Interpolate median value.
        com = np.sum(np.abs(hist) * np.arange(len(hist))) / np.sum(np.abs(hist))
        com_floor, com_ceil = int(np.floor(com)), int(np.ceil(com))
        median = bin_centres[com_floor] + (com_ceil - com) * (bin_centres[com_ceil] - bin_centres[com_floor])
    text = f"""\
mean: {mean:.2f}\n\
std: {std:.2f}\n\
median: {median:.2f}\n\
min/max: {min_val:.2f}/{max_val:.2f}\
"""
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.95, text, bbox=props, fontsize=fontsize, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right')

    if show:
        plt.show()

def plot_histograms(
    datas: Union[ImageData3D, List[ImageData3D]],
    axs: Optional[Union[mpl.axes.Axes, List[mpl.axes.Axes]]] = None,
    diffs: Optional[Union[ImageData3D, List[ImageData3D]]] = None,
    figsize: Tuple[float, float] = (6, 4),
    **kwargs) -> None:
    datas = arg_to_list(datas, (ImageData3D, str))
    diffs = arg_to_list(diffs, (ImageData3D, None), broadcast=len(datas))
    assert len(diffs) == len(datas)
    figsize = (len(datas) * figsize[0], figsize[1])
    if axs is None:
        _, axs = plt.subplots(1, len(datas), figsize=figsize, squeeze=False)
        axs = axs[0]    # Only one row.
        show = True
    else:
        axs = arg_to_list(axs, mpl.axes.Axes)
        assert len(axs) == len(datas), "Number of axes must match number of data arrays."
        show = False
    for a, d, diff in zip(axs, datas, diffs):
        plot_histogram(d, ax=a, diff=diff, **kwargs)

    if show:
        plt.show()

def plot_images(
    data: Union[ImageData3D, List[ImageData3D]],
    figsize: Tuple[float, float] = (16, 6),
    idxs: Union[int, float, List[Union[int, float]]] = 0.5,
    labels: Optional[Union[LabelData3D, List[Optional[LabelData3D]]]] = None,
    landmarks: Optional[Union[LandmarksData, List[LandmarksData]]] = None,    # Should be in patient coordinates.
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
            image, view_idx = __get_view_slice(d, idx, v)
            aspect = __get_aspect(v, s)
            origin = __get_origin(v)
            vmin, vmax = __get_window(window, d)
            if modality == 'ct':
                cmap='gray'
            elif modality == 'dose':
                cmap='viridis'
            col_ax.imshow(image, aspect=aspect, cmap=cmap, origin=origin, vmin=vmin, vmax=vmax)
            col_ax.set_title(f'{get_view_name(v)} view, slice {view_idx}')
            if l is not None:   # Plot landmarks.
                cmap = ListedColormap(((1, 1, 1, 0), palette[i]))
                label_image, _ = __get_view_slice(l, idx, v)
                col_ax.imshow(label_image, alpha=0.3, aspect=aspect, cmap=cmap, origin=origin)
                col_ax.contour(label_image, colors=[palette[i]], levels=[.5], linestyles='solid')
            if use_patient_coords:  # Change axis tick labels to show patient coordinates.
                size_x, size_y = __get_view_xy(d.shape, v)
                sx, sy = __get_view_xy(s, v)
                ox, oy = __get_view_xy(o, v)
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
                col_ax.set_title(f'{get_view_name(v)} view, slice {view_idx} ({view_loc:.1f}mm)')
            if lm is not None:
                # Convert landmarks to image coordinates.
                lmv = lm.copy()  # Stop overwriting of passed data.
                lmv[list(range(3))] = (lmv[list(range(3))] - o) / s
                __plot_landmarks_data(lmv, col_ax, view_idx, d.shape, spacing, offset, v)
            if p is not None:
                # Convert point to image coordinates.
                pv = (np.array(p) - o) / s
                px, py = __get_view_xy(pv, v)
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


def plot_heatmap(
    id: str,
    heatmap: np.ndarray,
    spacing: Spacing3D,
    alpha_heatmap: float = 0.7,
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[str] = None,
    crop: Optional[Union[str, PixelBox]] = None,
    crop_margin_mm: float = 100,
    ct_data: Optional[np.ndarray] = None,
    extent_of: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    fontsize: int = DEFAULT_FONT_SIZE,
    latex: bool = False,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    pred_data: Optional[Dict[str, np.ndarray]] = None,
    regions_data: Optional[Dict[str, np.ndarray]] = None,
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
        label = regions_data[centre] if type(centre) == str else centre
        extent_centre = centre_of_extent(label)
        idx = extent_centre[view]

    if extent_of is not None:
        # Get 'idx' at min/max extent of data.
        label = regions_data[extent_of[0]] if type(extent_of[0]) == str else extent_of
        extent_end = 0 if extent_of[1] == 'min' else 1
        ext = extent(label)
        idx = ext[extent_end][view]

    # Plot patient regions.
    size = heatmap.shape
    plot_patients(id, size, spacing, alpha_region=alpha_region, aspect=aspect, ax=ax, crop=crop, crop_margin_mm=crop_margin_mm, ct_data=ct_data, latex=latex, legend_loc=legend_loc, regions_data=regions_data, show=False, show_legend=False, idx=idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'PixelBox' type.
        if type(crop) == str:
            crop = __get_region_crop(regions_data[crop], crop_margin_mm, spacing, view)     # Crop was 'regions_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin_mm, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'PixelBox' type.

    # Get aspect ratio.
    if not aspect:
        aspect = __get_aspect(view, spacing) 

    # Get slice data.
    heatmap_slice, _ = __get_view_slice(heatmap, idx, view)

    # Crop the image.
    if crop is not None:
        heatmap_slice = crop_fn(heatmap_slice, __transpose_box(crop), use_patient_coords=False)

    # Plot heatmap
    image = ax.imshow(heatmap_slice, alpha=alpha_heatmap, aspect=aspect, origin=__get_origin(view))
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
                pred_slice, _ = __get_view_slice(pred, idx, view)

                # Crop the image.
                if crop:
                    pred_slice = crop_fn(pred_slice, __transpose_box(crop), use_patient_coords=False)

                # Plot prediction.
                if pred_slice.sum() != 0: 
                    cmap = ListedColormap(((1, 1, 1, 0), colour))
                    ax.imshow(pred_slice, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_origin(view))
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
        savepath = escape_filepath(savepath)
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

def plot_loaded(
    filepaths: Union[str, List[str]],
    figwidth: float = 16) -> None:
    filepaths = arg_to_list(filepaths, str)
    for f in filepaths:
        f = escape_filepath(f)
        images = convert_from_path(f)
        image = images[0]
        plt.figure(figsize=(figwidth, figwidth * image.height / image.width))
        plt.axis('off')
        plt.imshow(image)

def plot_patients(
    dataset_type: Dataset,
    dataset_fns: Dict[str, Callable],
    dataset: str,
    pat_ids: Union[PatientIDs, int] = 'all',
    centre: Optional[Union[LandmarkID, RegionID]] = None,
    crop: Optional[Union[str, PixelBox]] = None,
    isodoses: Union[float, List[float]] = [],
    landmark_ids: Optional[LandmarkIDs] = 'all',
    loadpaths: Union[str, List[str]] = [],
    modality: Optional[Union[DicomModality, NiftiModality, NrrdModality]] = None,    # Can be used instead of 'series_ids'.
    region_ids: Optional[RegionIDs] = 'all',
    region_labels: Dict[str, str] = {},
    series_ids: Optional[Union[SeriesID, List[SeriesID], Literal['all']]] = None,
    show_dose: bool = False,
    show_progress: bool = False,
    study_ids: Optional[Union[StudyID, List[StudyID], Literal['all']]] = None,
    **kwargs) -> None:
    isodoses = arg_to_list(isodoses, float)
    if len(loadpaths) > 0:
        plot_loaded(loadpaths)
        return

    # Get patient IDs.
    set = dataset_type(dataset)
    arg_pat_ids = pat_ids
    pat_ids = set.list_patients(pat_ids=pat_ids)
    if len(pat_ids) == 0:
        raise ValueError(f"No patients found for dataset '{dataset}' with IDs '{arg_pat_ids}'.")

    # Load all patient data.
    # This is the row-level data for 'plot_patients_matrix'.
    loaded_study_ids = []
    loaded_series_ids = []
    ct_datas = []
    dose_datas = []
    spacings = []
    offsets = []
    regions_datas = []
    landmarks_datas = []
    centre_datas = []
    crop_datas = []
    if show_progress:
        logging.info("Loading patient data...")
    for p in tqdm(pat_ids, disable=not show_progress):
        # Load single patient data.
        # This is the column-level data for 'plot_patients_matrix'.

        # Combinations:
        # one study ID, multiple series IDs (or all).
        # multiple study IDs (or all), one series IDs.

        # Get study ID/s.
        pat = set.patient(p)
        if study_ids is None:
            pat_study_ids = [pat.default_study.id]
        else:
            pat_study_ids = arg_to_list(study_ids, StudyID, literals={ 'all': pat.list_studies })

        row_series_ids = []
        row_study_ids = []
        row_ct_datas = []
        row_dose_datas = []
        row_spacings = []
        row_offsets = []
        row_regions_datas = []
        row_landmarks_datas = []
        row_centre_datas = []
        row_crop_datas = []
        for s in pat_study_ids:
            study = pat.study(s)
            if series_ids is not None:
                study_series_ids = arg_to_list(series_ids, (SeriesID, Union[DicomModality, NiftiModality]), literals={ 'all': study.list_series(('ct', 'mr')) })
                # Replace CT/MR with default series IDs.
                for i, ss in enumerate(study_series_ids):
                    if ss == 'ct':
                        study_series_ids[i] = study.default_ct.id
                    elif ss == 'mr':
                        study_series_ids[i] = study.default_mr.id
            elif modality is not None:
                study_series_ids = study.list_series(modality) if hasattr(study, 'list_series') else study.list_data(modality)
            else:
                study_series_ids = [study.default_ct.id if study.default_ct is not None else study.default_mr.id]

            # Add data for each series.
            for ss in study_series_ids:
                row_series_ids.append(ss)
                row_study_ids.append(s)

                # Load image data (e.g. CT/MR).
                ct_image = dataset_fns['ct_image'](study, ss)
                row_ct_datas.append(ct_image.data)
                row_spacings.append(ct_image.spacing)
                row_offsets.append(ct_image.offset)

                # Load region and landmark data from the same study.
                rdata = study.regions_data(region_ids=region_ids, **kwargs) if region_ids is not None else None
                row_regions_datas.append(rdata)
                ldata = study.landmarks_data(landmark_ids=landmark_ids) if landmark_ids is not None else None
                row_landmarks_datas.append(ldata)

                # Load dose data from the same study - and resample to image spacing.
                if (show_dose or len(isodoses) > 0) and dataset_fns['has_dose'](study):
                    dose_image = dataset_fns['dose_image'](study)
                    resample_kwargs = dict(
                        offset=dose_image.offset,
                        output_offset=ct_image.offset,
                        output_size=ct_image.size,
                        output_spacing=ct_image.spacing,
                        spacing=dose_image.spacing,
                    )
                    dose_data = resample(dose_image.data, **resample_kwargs)
                    row_dose_datas.append(dose_data)
                else:
                    row_dose_datas.append(None)

                # Handle centre.
                c = None
                if centre is not None:
                    if isinstance(centre, (LandmarkID, RegionID)):
                        # If data isn't in landmarks/regions_data then pass the data as 'centre', otherwise 'centre' can reference 
                        # the data in 'landmarks/regions_data'.
                        if study.has_landmarks(centre):
                            c = study.landmarks_data(landmark_ids=centre).iloc[0] if ldata is None or centre not in list(ldata['landmark-id']) else centre
                        elif study.has_regions(centre):
                            c = study.regions_data(region_ids=centre)[centre] if rdata is None or centre not in rdata else centre
                        else:
                            raise ValueError(f"Study {study} has no landmark/regions with ID '{centre}' for 'centre'.")
                row_centre_datas.append(c)

                # Add crop - load data if necessary.
                c = None
                if crop is not None:
                    if isinstance(crop, str):
                        if study.has_regions(crop):
                            if rdata is not None and crop in rdata:
                                c = crop    # Pass string, it will be read from 'regions_data' by 'plot_patient'.
                            else:
                                c = study.regions_data(region_ids=crop)[crop]  # Load RegionLabel.
                        elif study.has_landmarks(crop):
                            if ldata is not None and crop in list(ldata['landmark-id']):
                                c = crop
                            else:
                                c = study.landmarks_data(landmark_ids=crop).iloc[0]    # Load LandmarkData.
                row_crop_datas.append(c)

        # Apply region labels.
        # This should maybe be moved to base 'plot_patient'? All of the dataset-specific plotting functions
        # use this. Of course 'plot_patient' API would change to include 'region_labels' as an argument.
        for i in range(len(row_series_ids)):
            row_regions_datas[i], row_centre_datas[i], row_crop_datas[i] = __apply_region_labels(region_labels, row_regions_datas[i], row_centre_datas[i], row_crop_datas[i])

        loaded_study_ids.append(row_study_ids)
        loaded_series_ids.append(row_series_ids)
        ct_datas.append(row_ct_datas)
        dose_datas.append(row_dose_datas)
        spacings.append(row_spacings)
        offsets.append(row_offsets)
        regions_datas.append(row_regions_datas)
        landmarks_datas.append(row_landmarks_datas)
        centre_datas.append(row_centre_datas)
        crop_datas.append(row_crop_datas)

    # Plot.
    okwargs = dict(
        centres=centre_datas,
        crops=crop_datas,
        ct_datas=ct_datas,
        dose_datas=dose_datas,
        isodoses=isodoses,
        landmarks_datas=landmarks_datas,
        offsets=offsets,
        regions_datas=regions_datas,
        series_ids=loaded_series_ids,
        show_progress=show_progress,
        show_dose=show_dose,
        spacings=spacings,
        study_ids=loaded_study_ids,
    )
    plot_patients_matrix(pat_ids, **okwargs, **kwargs)

def plot_patients_matrix(
    # Allows us to plot multiple patients (rows) and patient studies, series, and views (columns).
    pat_ids: Union[str, List[str]],
    ax: Optional[mpl.axes.Axes] = None,
    centres: Optional[Union[LandmarkData, LandmarkID, RegionData, RegionID, List[Union[LandmarkData, LandmarkID, RegionData, RegionID]], List[Union[LandmarkData, LandmarkID, RegionData, RegionID, List[Union[LandmarkData, LandmarkID, RegionData, RegionID]]]]]] = None,
    crops: Optional[Union[str, np.ndarray, PixelBox, List[Union[str, np.ndarray, PixelBox]]]] = None,
    ct_datas: Optional[Union[CtData, List[CtData], List[Union[CtData, List[CtData]]]]] = None,
    dose_datas: Optional[Union[DoseData, List[DoseData], List[Union[DoseData, List[DoseData]]]]] = None,
    figsize: Tuple[int, int] = (46, 12),    # In cm.
    landmarks_datas: Optional[Union[LandmarksData, List[LandmarksData], List[Union[LandmarksData, List[LandmarksData]]]]] = None,
    offsets: Union[Point3D, List[Point3D], List[Union[Point3D, List[Point3D]]]] = None,
    regions_datas: Optional[Union[RegionsData, List[RegionsData], List[Union[RegionsData, List[RegionsData]]]]] = None,
    savepath: Optional[str] = None,
    series_ids: Union[StudyID, Sequence[StudyID], List[Union[StudyID, Sequence[StudyID]]]] = None,
    show_progress: bool = False,
    spacings: Union[Spacing3D, List[Spacing3D], List[Union[Spacing3D, List[Spacing3D]]]] = None,
    views: Union[Axis, List[Axis], Literal['all']] = 0,
    **kwargs) -> None:
    # Broadcast args to length of plot_ids.
    pat_ids = arg_to_list(pat_ids, PatientID)
    n_rows = len(pat_ids)
    spacings = arg_to_list(spacings, Spacing3D, broadcast=n_rows)
    centres = arg_to_list(centres, (None, LandmarkData, LandmarkID, RegionData, RegionID), broadcast=n_rows)
    crops = arg_to_list(crops, (None, LandmarkID, PixelBox, RegionID), broadcast=n_rows)
    ct_datas = arg_to_list(ct_datas, (None, CtData), broadcast=n_rows)
    dose_datas = arg_to_list(dose_datas, (None, DoseData), broadcast=n_rows)
    landmarks_datas = arg_to_list(landmarks_datas, (None, LandmarksData), broadcast=n_rows)
    regions_datas = arg_to_list(regions_datas, (None, RegionsData), broadcast=n_rows)
    views = arg_to_list(views, int, literals={ 'all': tuple(range(3)) })
    n_series_max = np.max([len(ss) for ss in series_ids])
    n_cols = len(views) * n_series_max

    # Convert figsize from cm to inches.
    figsize = __convert_figsize_to_inches(figsize)

    # Create axes.
    if ax is None:
        if n_rows > 1 or n_cols > 1:
            # Subplots for multiple views.
            _, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0], n_rows * figsize[1]), gridspec_kw={ 'hspace': 0.4 }, squeeze=False)
        else:
            plt.figure(figsize=figsize)
            ax = plt.axes(frameon=False)
            axs = [[ax]]
    else:
        axs = [[ax]]

    # Plot each plot.
    if show_progress:
        logging.info("Plotting patients...")
    for i in tqdm(range(n_rows), disable=not show_progress):
        pat_id = pat_ids[i]

        for j in range(n_cols):
            # Multiple studies can be provided for a patient.
            # Interleave studies and views across the columns.
            series_idx = j % n_series_max
            view_idx = j // n_series_max
            ct_data = ct_datas[i][series_idx] if isinstance(ct_datas[i], list) else ct_datas[i]
            dose_data = dose_datas[i][series_idx] if isinstance(dose_datas[i], list) else dose_datas[i]
            spacing = spacings[i][series_idx] if isinstance(spacings[i], list) else spacings[i]
            offset = offsets[i][series_idx] if isinstance(offsets[i], list) else offsets[i]
            regions_data = regions_datas[i][series_idx] if isinstance(regions_datas[i], list) else regions_datas[i]
            landmarks_data = landmarks_datas[i][series_idx] if isinstance(landmarks_datas[i], list) else landmarks_datas[i]
            crop = crops[i][series_idx] if isinstance(crops[i], list) else crops[i]
            centre = centres[i][series_idx] if isinstance(centres[i], list) else centres[i]
            view = views[view_idx] if len(views) > 1 else views[0]

            plot_patient(
                pat_id,
                ct_data.shape,
                spacing,
                ax=axs[i][j],
                centre=centre,
                close_figure=False,
                crop=crop,
                ct_data=ct_data,
                dose_data=dose_data,
                landmarks_data=landmarks_data,
                offset=offset,
                regions_data=regions_data,
                view=view,
                **kwargs)

    # Save plot to disk.
    if savepath is not None:
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if ax is None:
        plt.show()
        plt.close() 

def plot_patient(
    plot_id: str,
    size: Size3D,
    spacing: Spacing3D,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[Union[LandmarkData, LandmarkID, RegionData, RegionID]] = None,
    colours: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[LandmarkData, LandmarkID, PixelBox, RegionData, RegionID]] = None,    # Uses 'regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    crop_margin_mm: float = 100,                                       # Applied if cropping to 'regions_data' or 'np.ndarray'.
    crosshairs: Optional[Union[Pixel, Point2D, Landmark, LandmarkData]] = None,
    ct_data: Optional[CtData] = None,
    dose_alpha_min: float = 0.5,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'rainbow',
    dose_colourbar_pad: float = 0.05,
    dose_colourbar_size: float = 0.03,
    dose_data: Optional[DoseData] = None,
    escape_latex: bool = False,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'regions_data' if 'str', else 'np.ndarray') and min/max of extent.
    figsize: Tuple[float, float] = (36, 12),
    fontsize: int = DEFAULT_FONT_SIZE,
    idx: Optional[float] = None,
    idx_mm: Optional[float] = None,
    isodoses: Union[float, List[float]] = [],
    landmarks_data: Optional[LandmarksData] = None,     # All landmarks are plotted.
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    legend_show_all_regions: bool = False,
    linewidth: float = 0.5,
    linewidth_legend: float = 8,
    norm: Optional[Tuple[float, float]] = None,
    offset: Optional[Point3D] = None,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    regions_data: Optional[RegionsData] = None,         # All regions are plotted.
    savepath: Optional[str] = None,
    show_axes: bool = True,
    show_ct: bool = True,
    show_dose: bool = True,
    show_dose_legend: bool = True,
    show_extent: bool = False,
    show_legend: bool = False,
    show_title: bool = True,
    show_title_idx: bool = True,
    show_title_view: bool = True,
    show_x_label: bool = True,
    show_x_ticks: bool = True,
    show_y_label: bool = True,
    show_y_ticks: bool = True,
    title: Optional[str] = None,
    transform: torchio.transforms.Transform = None,
    use_patient_coords: bool = False,
    view: Axis = 0,
    window: Optional[Union[Literal['bone', 'lung', 'tissue'], Tuple[float, float]]] = 'tissue',
    window_mask: Optional[Tuple[float, float]] = None,
    **kwargs) -> None:
    if idx is None and idx_mm is None and centre is None and extent_of is None:
        idx = 0.5    # Default to middle of volume.
    __assert_view(view)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)
        show_figure = True
    else:
        show_figure = False

    # Convert to integer index in image coords.
    idx = __convert_idx(idx, size, view, idx_mm=idx_mm, spacing=spacing, offset=offset)

    # Get slice idx.
    if centre is not None:
        if isinstance(centre, (LandmarkID, RegionID)):
            if landmarks_data is not None and centre in list(landmarks_data['landmark-id']):
                centre_point = landmarks_data[landmarks_data['landmark-id'] == centre][list(range(3))].iloc[0]
                centre_vox = point_to_image_coords(centre_point, spacing, offset)
            elif regions_data is not None and centre in regions_data:
                centre_vox = __get_region_centre(regions_data[centre])
            else:
                raise ValueError(f"No centre '{centre}' found in 'landmarks/regions_data'.")
        elif isinstance(centre, LandmarkData):
            centre_point = tuple(landmarks_data[list(range(3))].iloc[0])
            centre_vox = point_to_image_coords(centre_point, spacing, offset)
        elif isinstance(centre, RegionData):
            centre_vox = __get_region_centre(centre)
        idx = centre_vox[view]

    # Get idx at min/max extent of label.
    if extent_of is not None:
        if len(extent_of) == 2:
            eo_region, eo_end = extent_of
            eo_axis = view
        elif len(extent_of) == 3:
            eo_region, eo_end, eo_axis = extent_of

        label = regions_data[eo_region] if type(eo_region) == str else eo_region     # 'eo_region' can be str ('regions_data' key) or np.ndarray.
        assert eo_end in ('min', 'max'), "'extent_of' must have one of ('min', 'max') as second element."
        eo_end = 0 if eo_end == 'min' else 1
        if postproc:
            label = postproc(label)
        ext_vox = extent(label, eo_axis, eo_end, view)
        idx = ext_vox[eo_end][axis]

    # Convert crop to PixelBox.
    crop_vox_xy = None
    if crop is not None:
        if isinstance(crop, str):
            if landmarks_data is not None and crop in list(landmarks_data['landmark-id']):
                lm_data = landmarks_data[landmarks_data['landmark-id'] == crop].iloc[0]
                crop_vox_xy = __get_landmarks_crop(lm_data, crop_margin_mm, size, spacing, offset, view)
            elif regions_data is not None and crop in regions_data:
                crop_vox_xy = __get_regions_crop(regions_data[crop], crop_margin_mm, spacing, offset, view)
        elif isinstance(crop, LandmarkData):
            crop_vox_xy = __get_landmarks_crop(crop, crop_margin_mm, size, spacing, offset, view)
        elif isinstance(centre, RegionData):
            crop_vox_xy = __get_regions_crop(crop, crop_margin_mm, spacing, offset, view)
        else:
            crop_vox_xy = tuple(*zip(crop))    # API accepts ((xmin, xmax), (ymin, ymax)) - convert to PixelBox.
            crop_vox_xy = replace_box_none(crop_vox_xy, size, use_patient_coords=False)
    else:
        crop_vox_xy = ((0, 0), __get_view_xy(size, view))  # Default to full image size.

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = __get_aspect(view, spacing) 

    # Plot CT data.
    if ct_data is not None and show_ct:
        # Plot CT slice.
        ct_slice, _ = __get_view_slice(ct_data, idx, view)
        ct_slice = crop_fn(ct_slice, __transpose_box(crop_vox_xy), use_patient_coords=False)
        vmin, vmax = __get_window(window=window, data=ct_data)
        ax.imshow(ct_slice, cmap='gray', aspect=aspect, interpolation='none', origin=__get_origin(view), vmin=vmin, vmax=vmax)

        # Highlight regions outside the window mask.
        if window_mask is not None:
            cmap = ListedColormap(((1, 1, 1, 0), 'red'))
            hw_slice = np.zeros_like(size)
            if window_mask[0] is not None:
                hw_slice[ct_slice < window_mask[0]] = 1
            if window_mask[1] is not None:
                hw_slice[ct_slice >= window_mask[1]] = 1
            ax.imshow(hw_slice, alpha=1.0, aspect=aspect, cmap=cmap, interpolation='none', origin=__get_origin(view))
    else:
        # Plot black background.
        empty_slice = np.zeros(__get_view_xy(size, view))
        empty_slice = crop_fn(empty_slice, __transpose_box(crop_vox_xy), use_patient_coords=False)
        ax.imshow(empty_slice, cmap='gray', aspect=aspect, interpolation='none', origin=__get_origin(view))

    # Plot crosshairs.
    if crosshairs is not None:
        # Convert crosshairs to Pixel.
        if isinstance(crosshairs, LandmarkID):
            if crosshairs not in list(landmarks_data['landmark-id']):
                raise ValueError(f"Landmark '{crosshairs}' not found in 'landmarks_data'.")
            lm_data = landmarks_data[landmarks_data['landmark-id'] == crosshairs]
            lm_data = landmarks_to_image_coords(lm_data, spacing, offset).iloc[0]
            crosshairs = __get_view_xy(lm_data[list(range(3))], view)
        elif use_patient_coords and isinstance(crosshairs, Point2D):
            # Passed crosshairs should be in same coordinates as image axes. Convert to image coords
            crosshairs = (np.array(crosshairs) - __get_view_xy(offset, view)) / __get_view_xy(spacing, view)

        crosshairs = np.array(crosshairs) - crop_vox_xy[0]
        ax.axvline(x=crosshairs[0], color='yellow', linewidth=linewidth, linestyle='dashed')
        ax.axhline(y=crosshairs[1], color='yellow', linewidth=linewidth, linestyle='dashed')
        ch_label = crosshairs.copy()
        ch_label = ch_label + crop_vox_xy[0]
        if use_patient_coords:
            ch_label = ch_label * __get_view_xy(spacing, view) + __get_view_xy(offset, view)
            ch_label = f'({ch_label[0]:.1f}, {ch_label[1]:.1f})'
        else:
            ch_label = f'({ch_label[0]}, {ch_label[1]})'
        ch_offset = 10
        ax.text(crosshairs[0] + ch_offset, crosshairs[1] - ch_offset, ch_label, fontsize=8, color='yellow')

    # Plot dose data.
    isodoses = arg_to_list(isodoses, float)
    if dose_data is not None and (show_dose or len(isodoses) > 0):
        dose_slice, _ = __get_view_slice(dose_data, idx, view)
        dose_slice = crop_fn(dose_slice, __transpose_box(crop_vox_xy), use_patient_coords=False)

        # Create colormap with varying alpha - so 0 Gray is transparent.
        mpl_cmap = plt.get_cmap(dose_cmap)
        dose_colours = mpl_cmap(np.arange(mpl_cmap.N))
        dose_colours[0, -1] = 0
        dose_colours[1:, -1] = np.linspace(dose_alpha_min, dose_alpha_max, mpl_cmap.N - 1)
        dose_cmap = ListedColormap(dose_colours)

        # Plot dose slice.
        if show_dose:
            imax = ax.imshow(dose_slice, aspect=aspect, cmap=dose_cmap, origin=__get_origin(view))
            if show_dose_legend:
                cbar = plt.colorbar(imax, fraction=dose_colourbar_size, pad=dose_colourbar_pad)
                cbar.set_label(label='Dose [Gray]', size=fontsize)
                cbar.ax.tick_params(labelsize=fontsize)

        # Plot isodose lines - use colourmap with alpha=1.
        if len(isodoses) > 0:
            dose_max = dose_data.max()
            isodose_levels = [d * dose_max for d in isodoses]
            dose_colours = mpl_cmap(np.arange(mpl_cmap.N))
            isodose_cmap = mpl.colors.ListedColormap(dose_colours)
            isodose_colours = [isodose_cmap(d) for d in isodoses]
            imax = ax.contour(dose_slice, colors=isodose_colours, levels=isodose_levels)

    # Plot landmarks.
    if landmarks_data is not None:
        __plot_landmarks_data(landmarks_data, ax, idx, size, spacing, offset, view, crop=crop_vox_xy, fontsize=fontsize, **kwargs)

    if regions_data is not None:
        # Plot regions.
        okwargs = dict(
            alpha=alpha_region,
            colours=colours,
            crop=crop_vox_xy,
            escape_latex=escape_latex,
            legend_show_all_regions=legend_show_all_regions,
            show_extent=show_extent,
            view=view,
        )
        should_show_legend = __plot_regions_data(regions_data, ax, idx, aspect, **okwargs)

        # Create legend.
        if show_legend and should_show_legend:
            plt_legend = ax.legend(bbox_to_anchor=legend_bbox_to_anchor, fontsize=fontsize, loc=legend_loc)
            frame = plt_legend.get_frame()
            frame.set_boxstyle('square', pad=0.1)
            frame.set_edgecolor('black')
            frame.set_linewidth(linewidth)
            for l in plt_legend.get_lines():
                l.set_linewidth(linewidth_legend)

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

        if use_patient_coords:
            ax.set_xlabel('mm')
        else:
            ax.set_xlabel(f'voxel [@ {spacing_x:.3f} mm]')

    if show_y_label:
        # Add 'y-axis' label.
        if view == 0:
            spacing_y = spacing[2]
        elif view == 1:
            spacing_y = spacing[2]
        elif view == 2:
            spacing_y = spacing[1]

        if use_patient_coords:
            ax.set_ylabel('mm')
        else:
            ax.set_ylabel(f'voxel [@ {spacing_y:.3f} mm]')

    # Show axis markers.
    if not show_x_ticks:
        ax.get_xaxis().set_ticks([])
    if not show_y_ticks:
        ax.get_yaxis().set_ticks([])

    # Adjust tick labels.
    axes_xs = [ax.get_xaxis(), ax.get_yaxis()]
    sizes_xy = __get_view_xy(size, view)
    crop_min_xy, crop_max_xy = crop_vox_xy
    spacing_xy = __get_view_xy(spacing, view)
    offset_xy = __get_view_xy(offset, view)
    for a, sp, o, cm, cx in zip(axes_xs, spacing_xy, offset_xy, crop_min_xy, crop_max_xy):
        # Ticks are in image coords, labels could be either image/patient coords.
        tick_spacing_vox = int(np.diff(a.get_ticklocs())[0])    # Use spacing from default layout.
        n_ticks = int(np.floor((cx - cm) / tick_spacing_vox)) + 1
        ticks = np.arange(n_ticks) * tick_spacing_vox
        if use_patient_coords:
            tick_labels = (ticks + cm) * sp + o
            tick_labels = [f'{l:.1f}' for l in tick_labels]
        else:
            tick_labels = ticks + cm
        a.set_ticks(ticks, labels=tick_labels)

    if show_title:
        # Add title.
        if title is None:
            # Set default title.
            n_slices = size[view]
            title = plot_id
            if show_title_idx:
                if use_patient_coords:
                    slice_mm = spacing[view] * idx + offset[view]
                    title = f"{title}, {slice_mm:.1f}mm"
                else:
                    title = f"{title}, {idx}/{n_slices - 1}"
            if show_title_view:
                title = f"{title} ({get_view_name(view)})"

        # Escape text if using latex.
        if escape_latex:
            title = __escape_latex(title)

        ax.set_title(title)

    # Save plot to disk.
    if savepath is not None:
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    if show_figure:
        plt.show()
        plt.close()

def plot_localiser_prediction(
    id: str,
    spacing: Spacing3D, 
    pred_data: np.ndarray,
    pred_region: str,
    aspect: float = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[str] = None,
    crop: PixelBox = None,
    crop_margin_mm: float = 100,
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
    regions_data: Optional[Dict[str, np.ndarray]] = None,
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
        label = regions_data[centre] if type(centre) == str else centre
        extent_centre = centre_of_extent(label)
        idx = extent_centre[view]

    if extent_of is not None:
        if len(extent_of) == 2:
            eo_region, eo_end = extent_of
            eo_axis = view
        elif len(extent_of) == 3:
            eo_region, eo_end, eo_axis = extent_of

        # Get 'slice' at min/max extent of data.
        label = regions_data[eo_region] if type(eo_region) == str else eo_region     # 'eo_region' can be str ('regions_data' key) or np.ndarray.
        assert eo_end in ('min', 'max'), "'extent_of' must have one of ('min', 'max') as second element."
        eo_end = 0 if eo_end == 'min' else 1
        if postproc:
            label = postproc(label)
        ext = extent(label)
        idx = ext[eo_end][axis]

    # Plot patient regions.
    plot_patients(id, pred_data.shape, spacing, aspect=aspect, ax=ax, crop=crop, ct_data=ct_data, figsize=figsize, escape_latex=escape_latex, legend_loc=legend_loc, regions_data=regions_data, show_legend=show_legend, show_extent=show_label_extent, idx=idx, view=view, **kwargs)

    if crop is not None:
        # Convert 'crop' to 'PixelBox' type.
        if type(crop) == str:
            crop = __get_region_crop(regions_data[crop], crop_margin_mm, spacing, view)     # Crop was 'regions_data' key.
        elif type(crop) == np.ndarray:
            crop = __get_region_crop(crop, crop_margin_mm, spacing, view)                  # Crop was 'np.ndarray'.
        else:
            crop = tuple(zip(*crop))                                                    # Crop was 'PixelBox' type.

    # Plot prediction.
    if show_pred and not empty_pred:
        # Get aspect ratio.
        if not aspect:
            aspect = __get_aspect(view, spacing) 

        # Get slice data.
        pred_slice, _ = __get_view_slice(pred_data, idx, view)

        # Crop the image.
        if crop:
            pred_slice = crop_fn(pred_slice, __transpose_box(crop), use_patient_coords=False)

        # Plot prediction.
        colours = [(1, 1, 1, 0), pred_colour]
        cmap = ListedColormap(colours)
        plt.imshow(pred_slice, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_origin(view))
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
        pred_centre = __get_view_xy(pred_centre, view)
            
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
        savepath = escape_filepath(savepath)
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
    spacing: Spacing3D,
    pred_data: RegionsData,
    alpha_diff: float = 0.7,
    alpha_pred: float = 0.5,
    alpha_region: float = 0.5,
    aspect: float = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[str] = None,
    crop: Optional[Union[str, PixelBox]] = None,
    ct_data: Optional[np.ndarray] = None,
    escape_latex: bool = False,
    extent_of: Optional[Tuple[str, Literal[0, 1]]] = None,
    figsize: Tuple[float, float] = (36, 12),
    fontsize: float = DEFAULT_FONT_SIZE,
    idx: Optional[int] = None,
    legend_bbox: Optional[Tuple[float, float]] = None,
    legend_loc: Union[str, Tuple[float, float]] = 'upper right',
    pred_colours: Optional[Union[str, List[str]]] = None,
    regions_data: Optional[RegionsData] = None,
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
    n_regions = len(regions_data.keys()) if regions_data is not None else 0

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
        label = regions_data[centre] if type(centre) == str else centre
        extent_centre = centre_of_extent(label)
        idx = extent_centre[view]

    # Get idx at min/max extent of data.
    if extent_of is not None:
        label = regions_data[extent_of[0]] if type(extent_of[0]) == str else extent_of
        extent_end = 0 if extent_of[1] == 'min' else 1
        ext = extent(label)
        idx = ext[extent_end][view]

    # Convert float idx to int.
    size = pred_data[list(pred_data.keys())[0]].shape
    if idx > 0 and idx < 1:
        idx = int(np.floor(idx * size[view]))

    # Plot patient regions - even if no 'ct_data/regions_data' we still want to plot shape as black background.
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
        regions_datas=regions_data,
        show=False,
        show_legend=False,
        idx=idx,
        views=view,
    )
    plot_patients(id, spacing, **okwargs, **kwargs)

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
    plot_patients(id, spacing, **okwargs, **kwargs)

    for i in range(n_pred_regions):
        region = pred_regions[i]
        pred = pred_data[region]
        colour = pred_colours[i]

        if pred.sum() != 0 and show_pred:
            # Get aspect ratio.
            if not aspect:
                aspect = __get_aspect(view, spacing) 

            # Get slice data.
            pred_slice, _ = __get_view_slice(pred, idx, view)

            # Crop the image.
            if crop:
                pred_slice = crop_fn(pred_slice, __transpose_box(crop), use_patient_coords=False)

            # Plot prediction.
            if pred_slice.sum() != 0: 
                cmap = ListedColormap(((1, 1, 1, 0), colour))
                axs[1].imshow(pred_slice, alpha=alpha_pred, aspect=aspect, cmap=cmap, origin=__get_origin(view))
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
    plot_patients(id, spacing, **okwargs, **kwargs)

    # Plot diffs.
    # Calculate single diff across all regions.
    label = np.zeros_like(regions_data[list(regions_data.keys())[0]])
    pred = np.zeros_like(pred_data[list(pred_data.keys())[0]])
    for i in range(n_pred_regions):
        region = pred_regions[i]
        label += regions_data[region]
        label = np.clip(label, a_min=0, a_max=1)  # In case over overlapping regions.
        pred += pred_data[region]
        pred = np.clip(pred, a_min=0, a_max=1)  # In case over overlapping regions.
    diff = pred - label

    if diff.sum() != 0:
        # Get aspect ratio.
        if not aspect:
            aspect = __get_aspect(view, spacing) 

        # Get slice data.
        diff_slice, _ = __get_view_slice(diff, idx, view)

        # Crop the image.
        if crop:
            diff_slice = crop_fn(diff_slice, __transpose_box(crop), use_patient_coords=False)

        # Plot prediction.
        cmap = ListedColormap(('red', (1, 1, 1, 0), 'green'))     # Red for false negatives, green for false positives.
        axs[2].imshow(diff_slice, alpha=alpha_diff, aspect=aspect, cmap=cmap, origin=__get_origin(view))
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
        savepath = escape_filepath(savepath)
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
    figsize: Tuple[float, float] = (16, 6),
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
    paired_by: Optional[Union[str, List[str]]] = None,
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
    if len(data) == 0:
        raise ValueError("DataFrame is empty.")
    hue_hatches = arg_to_list(hue_hatch, str)
    hue_labels = arg_to_list(hue_label, str)
    include_xs = arg_to_list(include_x, str)
    exclude_xs = arg_to_list(exclude_x, str)
    if show_hue_connections and hue_connections_index is None:
        raise ValueError(f"Please set 'hue_connections_index' to allow matching points between hues.")
    if show_stats and paired_by is None:
        raise ValueError(f"Please set 'paired_by' to determine sample pairing for Wilcoxon test.")
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
    if len(data) == 0:
        raise ValueError(f"DataFrame is empty after applying filters: {filt}.")
        
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
            _, axs = plt.subplots(n_rows, 1, constrained_layout=True, dpi=dpi, figsize=(figsize[0], n_rows * figsize[1]), gridspec_kw=gridspec_kw, sharex=True, sharey=share_y)
        else:
            plt.figure(dpi=dpi, figsize=figsize)
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
                    row_pivot_df = row_df.pivot(index=paired_by, columns=x, values=y).reset_index()
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
                            x_pivot_df = x_df.pivot(index=paired_by, columns=[hue], values=[y]).reset_index()
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
                            x_pivot_df = x_df.pivot(index=paired_by, columns=[hue], values=[y]).reset_index()
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
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', dpi=dpi, pad_inches=save_pad_inches)
        logging.info(f"Saved plot to '{savepath}'.")

def plot_patient_histograms(
    dataset_type: Dataset,
    dataset: str,
    pat_ids: PatientIDs = 'all',
    savepath: Optional[str] = None,
    show_progress: bool = False,
    study_ids: StudyIDs = 'all',
    **kwargs) -> None:
    set = dataset_type(dataset)
    pat_ids = set.list_patients(pat_ids=pat_ids)
    n_rows = len(pat_ids)

    # Get n_cols.
    n_cols = 0
    study_idses = []
    for p in pat_ids:
        pat = set.patient(p)
        ids = pat.list_studies(study_ids=study_ids)
        study_idses.append(ids)
        if len(ids) > n_cols:
            n_cols = len(ids)
    
    _, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    if show_progress:
        logging.info("Plotting patient histograms...")
    for i, p in tqdm(enumerate(pat_ids), disable=not show_progress):
        pat = set.patient(p)
        row_axs = axs[i]
        ss = study_idses[i]
        for col_ax, s in zip(row_axs, ss):
            study = pat.study(s)
            ct_data = study.ct_data.flatten()
            title = f"{p}:{s}"
            plot_histograms(ct_data, axs=col_ax, title=title, **kwargs)

    # Save plot to disk.
    if savepath is not None:
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    plt.show()

def plot_registrations(
    dataset_type: Dataset,
    load_reg_fn: Callable,
    dataset: str,
    model: str,
    centre: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[str, List[str]]] = None,
    crop_margin_mm: float = 100,
    exclude_fixed_pat_ids: Optional[PatientIDs] = None,
    exclude_moving_pat_ids: Optional[PatientIDs] = None,
    fixed_pat_ids: Optional[PatientIDs] = 'all',
    fixed_study_id: StudyID = 'study_1',
    idx: Optional[Union[int, float, List[Union[int, float]]]] = None,
    labels: Literal['included', 'excluded', 'all'] = 'all',
    landmarks: Optional[Landmarks] = 'all',
    loadpath: Optional[str] = None,
    moving_pat_ids: Optional[PatientIDs] = None,
    moving_study_id: StudyID = 'study_0',
    regions: Optional[Regions] = 'all',
    region_labels: Optional[Dict[str, str]] = None,
    show_dose: bool = False,
    splits: Splits = 'all',
    **kwargs) -> None:
    if centre is None and idx is None:
        idx = 0.5
    if loadpath is not None:
        plot_loaded(loadpath)
        return
    set = dataset_type(dataset)
    fixed_pat_ids = set.list_patients(exclude=exclude_fixed_pat_ids, pat_ids=fixed_pat_ids, splits=splits)
    if moving_pat_ids is None:
        moving_pat_ids = fixed_pat_ids
    else:
        moving_pat_ids = arg_to_list(moving_pat_ids, PatientID, literals={ 'all': set.list_patients(exclude=exclude_moving_pat_ids, splits=splits) })
        assert len(moving_pat_ids) == len(fixed_pat_ids)

    fixed_study_ids = arg_broadcast(fixed_study_id, len(fixed_pat_ids))
    moving_study_ids = arg_broadcast(moving_study_id, len(moving_pat_ids))

    moving_ct_datas, fixed_ct_datas, moved_ct_datas = [], [], []
    moving_dose_datas, fixed_dose_datas, moved_dose_datas = [], [], []
    moving_centres, fixed_centres, moved_centres = [], [], []
    moving_crops, fixed_crops, moved_crops = [], [], []
    moving_spacings, fixed_spacings, moved_spacings = [], [], []
    moving_offsets, fixed_offsets, moved_offsets = [], [], []
    moving_landmarks_datas, fixed_landmarks_datas, moved_landmarks_datas = [], [], []
    moving_regions_datas, fixed_regions_datas, moved_regions_datas = [], [], []
    moving_idxs, fixed_idxs, moved_idxs = [], [], []
    transforms = []

    for i, p in enumerate(fixed_pat_ids):
        moving_pat_id = p if moving_pat_ids is None else moving_pat_ids[i]

        # Load moving and fixed CT and region data.
        ids = [(moving_pat_id, moving_study_id), (p, fixed_study_id)]
        ct_datas = []
        dose_datas = []
        landmarks_datas = []
        regions_datas = []
        spacings = []
        centres = []
        crops = []
        offsets = []
        centres_broad = arg_broadcast(centre, 3)
        crops_broad = arg_broadcast(crop, 3)
        idxs_broad = arg_broadcast(idx, 3)
        for j, (p, s) in enumerate(ids):
            study = set.patient(p).study(s)
            ct_image = study.default_ct
            ct_datas.append(ct_image.data)
            spacings.append(ct_image.spacing)
            offsets.append(ct_image.offset)
            if show_dose and study.has_dose:
                # Resample dose data to CT spacing/offset.
                dose_image = study.default_dose
                resample_kwargs = dict(
                    offset=dose_image.offset,
                    output_offset=ct_image.offset,
                    output_size=ct_image.size,
                    output_spacing=ct_image.spacing,
                    spacing=dose_image.spacing,
                )
                dose_data = resample(dose_image.data, **resample_kwargs)
            else:
                dose_data = None
            dose_datas.append(dose_data)
            if landmarks is not None:
                landmarks_data = study.landmarks_data(landmarks=landmarks, use_patient_coords=False)
            else:
                landmarks_data = None
            if regions is not None:
                regions_data = study.regions_data(labels=labels, regions=regions)
            else:
                regions_data = None

            # Load 'centre' data if not already in 'regions_data'.
            centre = centres_broad[j]
            ocentre = None
            if centre is not None:
                if type(centre) == str:
                    if regions_data is None or centre not in regions_data:
                        ocentre = study.regions_data(regions=centre)[centre]
                    else:
                        ocentre = centre
                else:
                    ocentre = centre

            # Load 'crop' data if not already in 'regions_data'.
            crop = crops_broad[j]
            ocrop = None
            if crop is not None:
                if type(crop) == str:
                    if regions_data is None or crop not in regions_data:
                        ocrop = study.regions_data(regions=crop)[crop]
                    else:
                        ocrop = crop
                else:
                    ocrop = crop

            # Map region names.
            if region_labels is not None:
                # Rename regions.
                for o, n in region_labels.items():
                    regions_data[n] = regions_data.pop(o)

                # Rename 'centre' and 'crop' keys.
                if type(ocentre) == str and ocentre in region_labels:
                    ocentre = region_labels[ocentre] 
                if type(ocrop) == str and ocrop in region_labels:
                    ocrop = region_labels[ocrop]
            
            landmarks_datas.append(landmarks_data)
            regions_datas.append(regions_data)
            centres.append(ocentre)
            crops.append(ocrop)

        # Load registered data.
        transform, moved_ct_data, moved_regions_data, moved_landmarks_data, moved_dose_data = load_reg_fn(dataset, p, model, fixed_study_id=fixed_study_id, landmarks=landmarks, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, regions=regions, use_patient_coords=False) 

        # Load 'moved_centre' data if not already in 'moved_regions_data'.
        centre = centres_broad[2]
        moved_centre = None
        if centre is not None:
            if type(centre) == str:
                if moved_regions_data is None or centre not in moved_regions_data:
                    _, _, centre_regions_data, _, _ = load_reg_fn(dataset, p, model, fixed_study_id=fixed_study_id, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, regions=centre)
                    moved_centre = centre_regions_data[centre]
                else:
                    moved_centre = centre
            else:
                moved_centre = centre

        # Load 'moved_crop' data if not already in 'moved_regions_data'.
        crop = crops_broad[2]
        moved_crop = None
        if crop is not None:
            if type(crop) == str:
                if moved_regions_data is None or crop not in moved_regions_data:
                    _, _, crop_regions_data, _, _ = load_reg_fn(dataset, p, model, fixed_study_id=fixed_study_id, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, regions=crop)
                    moved_crop = crop_regions_data[crop]
                else:
                    moved_crop = crop
            else:
                moved_crop = crop

        # Rename moved labels.
        if region_labels is not None:
            # Rename regions.
            for o, n in region_labels.items():
                moved_regions_data[n] = moved_regions_data.pop(o)

            # Rename 'centre' and 'crop' keys.
            if type(moved_centre) == str and moved_centre in region_labels:
                moved_centre = region_labels[moved_centre] 
            if type(moved_crop) == str and moved_crop in region_labels:
                moved_crop = region_labels[moved_crop]

        # Add to main lists.
        fixed_ct_datas.append(ct_datas[1])
        moving_ct_datas.append(ct_datas[0])
        moved_ct_datas.append(moved_ct_data)
        fixed_spacings.append(spacings[1])
        moving_spacings.append(spacings[0])
        moved_spacings.append(spacings[1])
        fixed_offsets.append(offsets[1])
        moving_offsets.append(offsets[0])
        moved_offsets.append(offsets[1])
        fixed_dose_datas.append(dose_datas[1])
        moving_dose_datas.append(dose_datas[0])
        moved_dose_datas.append(moved_dose_data)
        fixed_centres.append(centres[1])
        moving_centres.append(centres[0])
        moved_centres.append(moved_centre)
        fixed_crops.append(crops[1])
        moving_crops.append(crops[0])
        moved_crops.append(moved_crop)
        fixed_landmarks_datas.append(landmarks_datas[1])
        moving_landmarks_datas.append(landmarks_datas[0])
        moved_landmarks_datas.append(moved_landmarks_data)
        fixed_regions_datas.append(regions_datas[1])
        moving_regions_datas.append(regions_datas[0])
        moved_regions_datas.append(moved_regions_data)
        fixed_idxs.append(idxs_broad[1])
        moving_idxs.append(idxs_broad[0])
        moved_idxs.append(idxs_broad[2])
        transforms.append(transform)

    okwargs = dict(
        fixed_centres=fixed_centres,
        fixed_crops=fixed_crops,
        fixed_crop_margin_mm=crop_margin_mm,
        fixed_ct_datas=fixed_ct_datas,
        fixed_dose_datas=fixed_dose_datas,
        fixed_idxs=fixed_idxs,
        fixed_landmarks_datas=fixed_landmarks_datas,
        fixed_offsets=fixed_offsets,
        fixed_spacings=fixed_spacings,
        fixed_regions_datas=fixed_regions_datas,
        moved_centres=moved_centres,
        moved_crops=moved_crops,
        moved_crop_margin_mm=crop_margin_mm,
        moved_ct_datas=moved_ct_datas,
        moved_dose_datas=moved_dose_datas,
        moved_idxs=moved_idxs,
        moved_landmarks_datas=moved_landmarks_datas,
        moved_regions_datas=moved_regions_datas,
        moving_centres=moving_centres,
        moving_crops=moving_crops,
        moving_crop_margin_mm=crop_margin_mm,
        moving_ct_datas=moving_ct_datas,
        moving_dose_datas=moving_dose_datas,
        moving_idxs=moving_idxs,
        moving_landmarks_datas=moving_landmarks_datas,
        moving_offsets=moving_offsets,
        moving_spacings=moving_spacings,
        moving_regions_datas=moving_regions_datas,
        transforms=transforms,
    )
    plot_registrations_matrix(fixed_pat_ids, fixed_study_ids, moving_pat_ids, moving_study_ids, **okwargs, **kwargs)

def plot_registrations_matrix(
    fixed_pat_ids: Sequence[PatientID],
    fixed_study_ids: Sequence[StudyID],
    moving_pat_ids: Sequence[PatientID],
    moving_study_ids: Sequence[StudyID],
    fixed_centres: Sequence[Optional[Union[str, np.ndarray]]] = [],             # Uses 'fixed_regions_data' if 'str', else uses 'np.ndarray'.
    fixed_crops: Sequence[Optional[Union[str, np.ndarray, PixelBox]]] = [],    # Uses 'fixed_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    fixed_ct_datas: Sequence[Optional[CtDatas]] = [],
    fixed_dose_datas: Sequence[Optional[DoseDatas]] = [],
    fixed_idxs: Sequence[Optional[Union[int, float]]] = [],
    fixed_landmarks_datas: Sequence[Optional[LandmarksData]] = [],
    fixed_offsets: Sequence[Optional[Point3D]] = [],
    fixed_regions_datas: Sequence[Optional[np.ndarray]] = [],
    fixed_spacings: Sequence[Optional[Spacing3D]] = [],
    figsize: Tuple[float, float] = (16, 4),     # Width always the same, height is based on a single row.
    moved_centres: Sequence[Optional[Union[str, LabelData3D]]] = [],             # Uses 'moved_regions_data' if 'str', else uses 'np.ndarray'.
    moved_crops: Sequence[Optional[Union[str, LabelData3D, PixelBox]]] = [],    # Uses 'moved_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moved_ct_datas: Sequence[Optional[CtDatas]] = [],
    moved_dose_datas: Sequence[Optional[DoseDatas]] = [],
    moved_idxs: Sequence[Optional[Union[int, float]]] = [],
    moved_landmarks_datas: Sequence[Optional[LandmarksData]] = [],
    moved_regions_datas: Sequence[Optional[RegionsData]] = [],
    moving_centres: Sequence[Optional[Union[str, LabelData3D]]] = [],             # Uses 'moving_regions_data' if 'str', else uses 'np.ndarray'.
    moving_crops: Sequence[Optional[Union[str, LabelData3D, PixelBox]]] = [],    # Uses 'moving_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moving_ct_datas: Sequence[Optional[CtDatas]] = [],
    moving_dose_datas: Sequence[Optional[DoseDatas]] = [],
    moving_idxs: Sequence[Optional[Union[int, float]]] = [],
    moving_landmarks_datas: Sequence[Optional[LandmarksData]] = [],
    moving_offsets: Sequence[Optional[Point3D]] = [],
    moving_regions_datas: Sequence[Optional[RegionsData]] = [],
    moving_spacings: Sequence[Optional[Spacing3D]] = [],
    savepath: Optional[str] = None,
    show_fixed: bool = True,
    show_grid: bool = True,
    show_moving: bool = True,
    transforms: Sequence[Optional[sitk.Transform]] = [],
    **kwargs) -> None:
    # Create subplots.
    n_pats = len(moving_pat_ids)
    n_rows = 2 if show_grid else 1
    n_cols = show_moving + show_fixed + 1
    figsize = (figsize[0], n_pats * n_rows * figsize[1])
    _, axs = plt.subplots(n_pats * n_rows, n_cols, figsize=figsize, squeeze=False)

    # Handle arguments.
    transforms = arg_to_list(transforms, (sitk.Transform, None), broadcast=n_pats)
    for i in range(n_pats):
        pat_axs = axs[n_rows * i: n_rows * (i + 1)]
        pat_axs = pat_axs.flatten()
        okwargs = dict(
            axs=pat_axs,
            fixed_centre=fixed_centres[i],
            fixed_crop=fixed_crops[i],
            fixed_ct_data=fixed_ct_datas[i],
            fixed_dose_data=fixed_dose_datas[i],
            fixed_idx=fixed_idxs[i],
            fixed_landmarks_data=fixed_landmarks_datas[i],
            fixed_offset=fixed_offsets[i],
            fixed_regions_data=fixed_regions_datas[i],
            fixed_spacing=fixed_spacings[i],
            moved_centre=moved_centres[i],
            moved_crop=moved_crops[i],
            moved_ct_data=moved_ct_datas[i],
            moved_dose_data=moved_dose_datas[i],
            moved_idx=moved_idxs[i],
            moved_landmarks_data=moved_landmarks_datas[i],
            moved_regions_data=moved_regions_datas[i],
            moved_spacing=moving_spacings[i],
            moving_centre=moving_centres[i],
            moving_crop=moving_crops[i],
            moving_ct_data=moving_ct_datas[i],
            moving_dose_data=moving_dose_datas[i],
            moving_idx=moving_idxs[i],
            moving_landmarks_data=moving_landmarks_datas[i],
            moving_offset=moving_offsets[i],
            moving_regions_data=moving_regions_datas[i],
            moving_spacing=moving_spacings[i],
            show_fixed=show_fixed,
            show_grid=show_grid,
            show_moving=show_moving,
            transform=transforms[i],
        )
        plot_registration(fixed_pat_ids[i], fixed_study_ids[i], moving_pat_ids[i], moving_study_ids[i], **okwargs, **kwargs)

    # Save plot to disk.
    if savepath is not None:
        savepath = escape_filepath(savepath)
        dirpath = os.path.dirname(savepath)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        logging.info(f"Saved plot to '{savepath}'.")

    plt.show()
    plt.close()

def plot_registration(
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    axs: Optional[mpl.axes.Axes] = None,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'regions_data' if 'str', else 'np.ndarray') and min/max of extent.
    fixed_centre: Optional[Union[str, np.ndarray]] = None,             # Uses 'fixed_regions_data' if 'str', else uses 'np.ndarray'.
    fixed_crop: Optional[Union[str, np.ndarray, PixelBox]] = None,    # Uses 'fixed_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    fixed_crop_margin_mm: float = 100,                                       # Applied if cropping to 'fixed_regions_data' or 'np.ndarray'
    fixed_ct_data: Optional[CtData] = None,
    fixed_dose_data: Optional[DoseData] = None,
    fixed_idx: Optional[int] = None,
    fixed_landmarks_data: Optional[LandmarksData] = None,
    fixed_offset: Optional[Point3D] = None,
    fixed_regions_data: Optional[np.ndarray] = None,
    fixed_spacing: Optional[Spacing3D] = None,
    figsize: Tuple[float, float] = (30, 10),
    fontsize: int = DEFAULT_FONT_SIZE,
    latex: bool = False,
    match_landmarks: bool = True,
    moved_centre: Optional[Union[str, np.ndarray]] = None,             # Uses 'moved_regions_data' if 'str', else uses 'np.ndarray'.
    moved_colour: str = 'red',
    moved_crop: Optional[Union[str, np.ndarray, PixelBox]] = None,    # Uses 'moved_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moved_crop_margin_mm: float = 100,                                       # Applied if cropping to 'moved_regions_data' or 'np.ndarray'
    moved_ct_data: Optional[CtData] = None,
    moved_dose_data: Optional[DoseData] = None,
    moved_idx: Optional[int] = None,
    moved_landmarks_data: Optional[Landmarks] = None,
    moved_regions_data: Optional[np.ndarray] = None,
    moved_use_fixed_idx: bool = True,
    moving_centre: Optional[Union[str, np.ndarray]] = None,             # Uses 'moving_regions_data' if 'str', else uses 'np.ndarray'.
    moving_crop: Optional[Union[str, np.ndarray, PixelBox]] = None,    # Uses 'moving_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moving_crop_margin_mm: float = 100,                                       # Applied if cropping to 'moving_regions_data' or 'np.ndarray'
    moving_ct_data: Optional[CtData] = None,
    moving_dose_data: Optional[DoseData] = None,
    moving_idx: Optional[int] = None,
    moving_landmarks_data: Optional[Landmarks] = None,
    moving_offset: Optional[Point3D] = None,
    moving_regions_data: Optional[np.ndarray] = None,
    moving_spacing: Optional[Spacing3D] = None,
    n_landmarks: Optional[int] = None,
    show_fixed: bool = True,
    show_grid: bool = True,
    show_legend: bool = False,
    show_moved_landmarks: bool = True,
    show_moving: bool = True,
    show_moving_landmarks: bool = True,
    show_region_overlay: bool = True,
    transform: Optional[Union[itk.Transform, sitk.Transform]] = None,
    transform_format: Literal['itk', 'sitk'] = 'sitk',
    view: Axis = 0,
    **kwargs) -> None:
    __assert_idx(fixed_centre, extent_of, fixed_idx)
    __assert_idx(moving_centre, extent_of, moving_idx)
    __assert_idx(moved_centre, extent_of, moved_idx)

    if n_landmarks is not None and match_landmarks:
        # Ensure that the n "moving/moved" landmarks targeted by "n_landmarks" are
        # are the same as the n "fixed" landmarks. 

        # Get n fixed landmarks.
        fixed_idx = __convert_idx(fixed_idx, fixed_ct_data.shape, view)
        fixed_landmarks_data['dist'] = np.abs(fixed_landmarks_data[view] - fixed_idx)
        fixed_landmarks_data = fixed_landmarks_data.sort_values('dist')
        fixed_landmarks_data = fixed_landmarks_data.iloc[:n_landmarks]
        fixed_landmarks = fixed_landmarks_data['landmark-id'].tolist()

        # Get moving/moved landmarks.
        moving_landmarks_data = moving_landmarks_data[moving_landmarks_data['landmark-id'].isin(fixed_landmarks)]
        moved_landmarks_data = moved_landmarks_data[moved_landmarks_data['landmark-id'].isin(fixed_landmarks)]

        okwargs = dict(
            colour=moved_colour,
            n_landmarks=n_landmarks, 
            zorder=0,  # Plot underneath "moving" (ground truth) landmarks.
        )

    # Get all plot parameters.
    hidden = 'HIDDEN'
    ct_datas = [moving_ct_data if show_moving else hidden, fixed_ct_data if show_fixed else hidden, moved_ct_data]
    ct_datas = [c for c in ct_datas if not (isinstance(c, str) and c == hidden)]
    dose_datas = [moving_dose_data if show_moving else hidden, fixed_dose_data if show_fixed else hidden, moved_dose_data]
    dose_datas = [c for c in dose_datas if not (isinstance(c, str) and c == hidden)]
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
    crop_margin_mms = [moving_crop_margin_mm if show_moving else hidden, fixed_crop_margin_mm if show_fixed else hidden, moved_crop_margin_mm]
    crop_margin_mms = [c for c in crop_margin_mms if not (isinstance(c, str) and c == hidden)]
    ids = [f'{moving_pat_id}:{moving_study_id}' if show_moving else hidden, f'{fixed_pat_id}:{fixed_study_id}' if show_fixed else hidden, f'moved']
    ids = [c for c in ids if not (isinstance(c, str) and c == hidden)]
    landmarks_datas = [moving_landmarks_data if show_moving else hidden, fixed_landmarks_data if show_fixed else hidden, None]
    if not show_moving_landmarks:
        # Hide moving landmarks, but we still need 'moving_landmarks_data' != None for the
        # "moved" landmark plotting code.
        landmarks_datas[0] = None
    landmarks_datas = [l for l in landmarks_datas if not (isinstance(l, str) and l == hidden)]
    regions_datas = [moving_regions_data if show_moving else hidden, fixed_regions_data if show_fixed else hidden, moved_regions_data]
    regions_datas = [c for c in regions_datas if not (isinstance(c, str) and c == hidden)]
    idxs = [moving_idx if show_moving else hidden, fixed_idx if show_fixed else hidden, moved_idx]
    idxs = [c for c in idxs if not (isinstance(c, str) and c == hidden)]
    infos = [{} if show_moving else hidden, {} if show_fixed else hidden, { 'fixed': moved_use_fixed_idx }]
    infos = [c for c in infos if not (isinstance(c, str) and c == hidden)]

    n_rows = 2 if show_grid else 1
    n_cols = show_moving + show_fixed + 1
    axs = arg_to_list(axs, mpl.axes.Axes)
    if axs is None:
        figsize_width, figsize_height = figsize
        figsize_height = n_rows * figsize_height
        figsize = (figsize_width, figsize_height)
        # How do I remove vertical spacing???
        fig = plt.figure(figsize=__convert_figsize_to_inches(figsize))
        gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.05, wspace=0.3)
        axs = []
        # Add first row of images.
        for i in range(n_cols):
            ax = fig.add_subplot(gs[0, i])
            axs.append(ax)

        # Add second row of images.
        if show_grid or show_region_overlay:
            for i in range(n_cols):
                ax = fig.add_subplot(gs[1, i])
                axs.append(ax)

            # Remove axes from second row if necessary.
            if not show_region_overlay:
                axs[2 * n_cols - 2].remove()

        show_figure = True
    else:
        assert len(axs) == n_rows * n_cols, f"Expected {n_rows * n_cols} axes, but got {len(axs)}."
        show_figure = False

    # Set latex as text compiler.
    if latex:
        plt.rcParams.update({
            "font.family": "serif",
            'text.usetex': True
        })

    # Plot images.
    for i in range(n_cols):
        okwargs = dict(
            ax=axs[i],
            centre=centres[i],
            crop=crops[i],
            ct_data=ct_datas[i],
            dose_data=dose_datas[i],
            idx=idxs[i],
            landmarks_data=landmarks_datas[i],
            n_landmarks=n_landmarks,
            regions_data=regions_datas[i],
            show_legend=show_legend,
            view=view,
        )
        plot_patient(ids[i], ct_datas[i].shape, spacings[i], **okwargs, **kwargs)

    # Add moved landmarks.
    if moved_landmarks_data is not None and show_moved_landmarks:
        okwargs = dict(
            colour=moved_colour,
            zorder=0,
        )
        __plot_landmarks_data(moved_landmarks_data, axs[0], idxs[0], sizes[0], spacing, offset, view, fontsize=fontsize, **okwargs, **kwargs)

    if show_grid:
        # Plot moving grid.
        include = [True] * 3
        include[view] = False
        moving_grid = __create_grid(moving_ct_data.shape, moving_spacing, include=include)
        moving_idx = __convert_idx(moving_idx, moving_ct_data.shape, view)
        if show_moving:
            grid_slice, _ = __get_view_slice(moving_grid, moving_idx, view)
            aspect = __get_aspect(view, moving_spacing)
            origin = __get_origin(view)
            axs[n_cols].imshow(grid_slice, aspect=aspect, cmap='gray', origin=origin)

        # Plot moved grid.
        moved_idx = __convert_idx(moved_idx, moved_ct_data.shape, view)
        if transform_format == 'itk':
            # When ITK loads nifti images, it reversed direction/offset for x/y axes.
            # This is an issue as our code doesn't use directions, it assumes a positive direction matrix.
            # I don't know how to reverse x/y axes with ITK transforms, so we have to do it with 
            # images before applying the transform.
            moved_grid = itk_transform_image(moving_grid, transform, fixed_ct_data.shape, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing)
        elif transform_format == 'sitk':
            moved_grid = resample(moving_grid, offset=moving_offset, output_offset=fixed_offset, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
        grid_slice, _ = __get_view_slice(moved_grid, moving_idx, view)
        aspect = __get_aspect(view, fixed_spacing)
        origin = __get_origin(view)
        axs[2 * n_cols - 1].imshow(grid_slice, aspect=aspect, cmap='gray', origin=origin)

        if show_region_overlay:
            # Plot fixed/moved regions.
            aspect = __get_aspect(view, fixed_spacing)
            okwargs = dict(
                alpha=alpha_region,
                crop=fixed_crop,
                view=view,
            )
            f_idx = __convert_idx(fixed_idx, fixed_ct_data.shape, view)
            background, _ = __get_view_slice(np.zeros(shape=fixed_ct_data.shape), f_idx, view)
            if fixed_crop is not None:
                background = crop_fn(background, __transpose_box(fixed_crop), use_patient_coords=False)
            axs[2 * n_cols - 2].imshow(background, cmap='gray', aspect=aspect, interpolation='none', origin=__get_origin(view))
            if fixed_regions_data is not None:
                __plot_regions_data(fixed_regions_data, axs[2 * n_cols - 2], f_idx, aspect, **okwargs)
            if moved_regions_data is not None:
                __plot_regions_data(moved_regions_data, axs[2 * n_cols - 2], f_idx, aspect, **okwargs)

    if show_figure:
        plt.show()
        plt.close()

def __add_outlier_info(df, x, y, hue):
    if hue is not None:
        groupby = [hue, x]
    else:
        groupby = x
    q1_map = df.groupby(groupby)[y].quantile(.25)
    q3_map = df.groupby(groupby)[y].quantile(.75)
    def q_func_build(qmap):
        def q_func(row):
            if isinstance(groupby, list):
                key = tuple(row[groupby])
            else:
                key = row[groupby]
            return qmap[key]
        return q_func
    q1 = df.apply(q_func_build(q1_map), axis=1)
    df = df.assign(q1=q1)
    df = df.assign(q3=df.apply(q_func_build(q3_map), axis=1))
    df = df.assign(iqr=df.q3 - df.q1)
    df = df.assign(outlier_lim_low=df.q1 - 1.5 * df.iqr)
    df = df.assign(outlier_lim_high=df.q3 + 1.5 * df.iqr)
    df = df.assign(outlier=(df[y] < df.outlier_lim_low) | (df[y] > df.outlier_lim_high))
    return df

def __apply_region_labels(
    region_labels: Dict[str, str],
    regions_data: Optional[Dict[str, np.ndarray]],
    centre: Optional[str],
    crop: Optional[str]) -> Tuple[Dict[str, np.ndarray], Optional[str], Optional[str]]:

    # Apply region labels to 'regions_data' and 'centre/crop' keys.
    if regions_data is not None:
        for old, new in region_labels.items():
            regions_data[new] = regions_data.pop(old)
    if centre is not None and type(centre) == str and centre in region_labels:
        centre = region_labels[centre] 
    if centre is not None and type(crop) == str and crop in region_labels:
        crop = region_labels[crop]

    return regions_data, centre, crop

def __assert_dataframe(data: pd.DataFrame) -> None:
    if len(data) == 0:
        raise ValueError("Dataframe is empty.")

def __assert_data_type(
    name: str,
    data: Union[ImageData3D, Dict[str, np.ndarray]],
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
    if view not in (0, 1, 2):
        raise ValueError(f"View '{view}' not recognised. Must be one of (0, 1, 2).")

def __box_in_plane(
    box: VoxelBox,
    view: Axis,
    idx: int) -> bool:
    # Get view bounding box.
    min, max = box
    min = min[view]
    max = max[view]

    # Calculate if the box is in plane.
    result = idx >= min and idx <= max
    return result

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

def __convert_figsize_to_inches(figsize: Tuple[float, float]) -> Tuple[float, float]:
    cm_to_inch = 1 / 2.54
    figsize = figsize[0] * cm_to_inch if figsize[0] is not None else None, figsize[1] * cm_to_inch if figsize[1] is not None else None
    return figsize

def __convert_idx(
    idx: Optional[Union[int, float]],
    size: Size3D,
    view: Axis,
    idx_mm: Optional[float] = None,
    offset: Optional[Point3D] = None,
    spacing: Optional[Spacing3D] = None) -> int:
    # Map float idx (\in [0, 1]) to int idx (\in [0, size[view]]).
    if idx is not None and idx > 0 and idx < 1:
        idx = int(np.round(idx * (size[view] - 1)))
    elif idx_mm is not None:
        # Find nearest voxel index to mm position.
        idx = int(np.round((idx_mm - offset[view]) / spacing[view]))
    return idx

def __create_grid(
    size: Size3D,
    spacing: Spacing3D,
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

def __get_aspect(
    view: Axis,
    spacing: Spacing3D) -> float:
    if view == 0:
        aspect = spacing[2] / spacing[1]
    elif view == 1:
        aspect = spacing[2] / spacing[0]
    elif view == 2:
        aspect = spacing[1] / spacing[0]
    return np.abs(aspect)

def __get_landmarks_crop(
    landmark: LandmarkData,
    margin_mm: float,
    size: Size3D,
    spacing: Spacing3D,
    offset: Point3D,
    view: Axis) -> PixelBox:
    # Add crop margin.
    landmark = landmark[range(3)].to_numpy()
    min_mm = landmark - margin_mm
    max_mm = landmark + margin_mm

    # Convert to image coords.
    min_vox = tuple(np.floor((min_mm - offset) / spacing).astype(int))
    max_vox = tuple(np.ceil((max_mm - offset) / spacing).astype(int))

    # Don't pad original image.
    min_vox = tuple(int(v) for v in np.clip(min_vox, a_min=0, a_max=None))
    max_vox = tuple(int(v) for v in np.clip(max_vox, a_min=None, a_max=size))

    # Select 2D component.
    min_vox = __get_view_xy(min_vox, view)
    max_vox = __get_view_xy(max_vox, view)

    return min_vox, max_vox

def __get_origin(view: Axis) -> str:
    if view == 2:
        return 'upper'
    else:
        return 'lower'

def __get_region_centre(data: RegionData) -> int:
    if data.sum() == 0:
        raise ValueError("Centre region has no foreground voxels.")
    return centre_of_extent(data)

def __get_regions_crop(
    data: np.ndarray,
    crop_margin_mm: float,
    spacing: Spacing3D,
    offset: Point3D,
    view: Axis) -> PixelBox:
    # Get region extent.
    ext_min_mm, ext_max_mm = get_extent(data, spacing, offset)

    # Add crop margin.
    min_mm = np.array(ext_min_mm) - margin_mm
    max_mm = np.array(ext_max_mm) + margin_mm

    # Convert to image coords.
    min_vox = tuple(np.floor((min_mm - offset) / spacing).astype(int))
    max_vox = tuple(np.ceil((max_mm - offset) / spacing).astype(int))

    # Don't pad original image.
    min_vox = tuple(int(v) for v in np.clip(min_vox, a_min=0, a_max=None))
    max_vox = tuple(int(v) for v in np.clip(max_vox, a_min=None, a_max=size))

    # Select 2D component.
    min_vox = __get_view_xy(min_vox, view)
    max_vox = __get_view_xy(max_vox, view)

    return min_vox, max_vox

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

def __get_view_slice(
    data: Union[ImageData3D, VectorData],
    idx: Union[int, float],
    view: Axis) -> Tuple[ImageData2D, int]:
    n_dims = len(data.shape)
    if n_dims == 4:
        assert data.shape[0] == 3   # vector image.
        view_idx = view + 1
    else:
        view_idx = view

    # Check that slice index isn't too large.
    if idx >= data.shape[view_idx]:
        raise ValueError(f"Idx '{idx}' out of bounds, only '{data.shape[view_idx]}' {get_view_name(view)} indices.")
    
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

def __get_view_xy(
    data: Union[Spacing3D, Point3D],
    view: Axis) -> Tuple[float, float]:
    if view == 0:
        res = (data[1], data[2])
    elif view == 1:
        res = (data[0], data[2])
    elif view == 2:
        res = (data[0], data[1])
    return res

def __get_window(
    window: Optional[Union[str, Tuple[Optional[float], Optional[float]]]] = None,
    data: Optional[ImageData3D] = None) -> Tuple[float, float]:
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

def __plot_box_slice(
    box: VoxelBox,
    view: Axis,
    ax: Optional[mpl.axes.Axes] = None,
    colour: str = 'r',
    crop: PixelBox = None,
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

def __plot_landmarks_data(
    landmarks_data: LandmarksData,
    ax: mpl.axes.Axes,
    idx: Union[int, float],
    size: Size3D,
    spacing: Spacing3D,
    offset: Point3D,
    view: Axis, 
    colour: str = 'yellow',
    crop: Optional[PixelBox] = None,
    fontsize: float = 12,
    n_landmarks: Optional[int] = None,
    show_landmark_ids: bool = False,
    show_landmark_dists: bool = True,
    zorder: float = 1,
    **kwargs) -> None:
    idx = __convert_idx(idx, size, view)
    landmarks_data = landmarks_data.copy()

    # Convert landmarks to image coords.
    landmarks_data = landmarks_to_image_coords(landmarks_data, spacing, offset)

    # Take subset of n closest landmarks landmarks.
    landmarks_data['dist'] = np.abs(landmarks_data[view] - idx)
    if n_landmarks is not None:
        landmarks_data = landmarks_data.sort_values('dist')
        landmarks_data = landmarks_data.iloc[:n_landmarks]

    # Convert distance to alpha.
    dist_norm = landmarks_data['dist'] / (size[view] / 2)
    dist_norm[dist_norm > 1] = 1
    landmarks_data['dist-norm'] = dist_norm
    base_alpha = 0.3
    landmarks_data['alpha'] = base_alpha + (1 - base_alpha) * (1 - landmarks_data['dist-norm'])
    
    r, g, b = mpl.colors.to_rgb(colour)
    if show_landmark_dists:
        colours = [(r, g, b, a) for a in landmarks_data['alpha']]
    else:
        colours = [(r, g, b, 1) for _ in landmarks_data['alpha']]
    img_axs = list(range(3))
    img_axs.remove(view)
    lm_x, lm_y, lm_ids = landmarks_data[img_axs[0]], landmarks_data[img_axs[1]], landmarks_data['landmark-id']
    if crop is not None:
        lm_x, lm_y = lm_x - crop[0][0], lm_y - crop[0][1]
    ax.scatter(lm_x, lm_y, c=colours, s=20, zorder=zorder)
    if show_landmark_ids:
        for x, y, t in zip(lm_x, lm_y, lm_ids):
            ax.text(x, y, t, fontsize=fontsize, color='red')

def __plot_regions_data(
    data: RegionsData,
    ax: mpl.axes.Axes,
    idx: int,
    aspect: float,
    alpha: float = 0.3,
    colours: Optional[Union[str, List[str]]] = None,
    crop: Optional[PixelBox] = None,
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
        slice_data, _ = __get_view_slice(data[region], idx, view)

        # Crop image.
        if crop:
            slice_data = crop_fn(slice_data, __transpose_box(crop), use_patient_coords=False)

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
        ax.imshow(slice_data, alpha=alpha, aspect=aspect, cmap=cmap, interpolation='none', origin=__get_origin(view))
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

def __transpose_box(box: Union[PixelBox, VoxelBox]) -> Union[PixelBox, VoxelBox]:
    return tuple(tuple(reversed(b)) for b in box)
    
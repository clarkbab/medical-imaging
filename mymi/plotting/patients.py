import matplotlib as mpl
import torchio
from typing import *

from mymi.constants import *
from mymi.datasets import DicomModality, NiftiModality
from mymi.geometry import foreground_fov
from mymi.typing import *
from mymi.utils import *

from .data import plot_histograms
from .plotting import *

@alias_kwargs(('upc', 'use_patient_coords'))
def plot_single_patient(
    size: Size3D,
    spacing: Spacing3D,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[Union[LandmarkSeries, LandmarkID, Literal['dose'], RegionArray, RegionID]] = None,
    centre_other: Optional[Union[LandmarkSeries, LandmarkID, Literal['dose'], RegionArray, RegionID]] = None,
    colours: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[LandmarkSeries, LandmarkID, Box2D, RegionArray, RegionID]] = None,    # Uses 'regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    crop_margin_mm: float = 100,                                       # Applied if cropping to 'regions_data' or 'np.ndarray'.
    crosshairs: Optional[Union[LandmarkID, LandmarkSeries, Pixel, Point2D]] = None,
    ct_data: Optional[CtImageArray] = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    dose_colourbar_pad: float = 0.05,
    dose_colourbar_size: float = 0.03,
    dose_data: Optional[DoseImageArray] = None,
    dose_series: Optional[SeriesID] = None,
    dose_series_date: Optional[str] = None,
    escape_latex: bool = False,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'regions_data' if 'str', else 'np.ndarray') and min/max of extent.
    figsize: Tuple[float, float] = (36, 12),
    fontsize: int = 12,
    idx: Optional[float] = None,
    idx_mm: Optional[float] = None,
    isodoses: Union[float, List[float]] = [],
    landmarks_data: Optional[LandmarksFrame] = None,     # All landmarks are plotted.
    landmarks_data_other: Optional[LandmarksFrame] = None,     # Plotted as 'red' landmarks, e.g. from registration.
    landmark_series: Optional[SeriesID] = None,
    landmark_series_date: Optional[str] = None,
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = (1, 1),
    legend_loc: Union[str, Tuple[float, float]] = 'upper left',
    legend_show_all_regions: bool = False,
    linewidth: float = 0.5,
    linewidth_legend: float = 8,
    norm: Optional[Tuple[float, float]] = None,
    origin: Optional[Point3D] = None,
    pat_id: Optional[PatientID] = None,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    regions_data: Optional[RegionArrays] = None,         # All regions are plotted.
    region_series: Optional[SeriesID] = None,
    region_series_date: Optional[str] = None,
    savepath: Optional[str] = None,
    series: Optional[SeriesID] = None,
    series_date: Optional[str] = None,
    show_axes: bool = True,
    show_ct: bool = True,
    show_dose: bool = False,
    show_dose_legend: bool = True,
    show_extent: bool = False,
    show_legend: bool = False,
    show_title: bool = True,
    show_title_dose: bool = True,
    show_title_pat: bool = True,
    show_title_image_series: bool = True,
    show_title_landmark_series: bool = True,
    show_title_region_series: bool = True,
    show_title_slice: bool = True,
    show_title_study: bool = True,
    show_x_label: bool = True,
    show_x_ticks: bool = True,
    show_y_label: bool = True,
    show_y_ticks: bool = True,
    study_date: Optional[str] = None,
    study: Optional[StudyID] = None,
    title: Optional[str] = None,
    title_width: int = 20,
    transform: torchio.transforms.Transform = None,
    use_patient_coords: bool = True,
    view: Axis = 0,
    window: Optional[Union[Literal['bone', 'lung', 'tissue'], Tuple[float, float]]] = 'tissue',
    window_mask: Optional[Tuple[float, float]] = None,
    **kwargs) -> None:

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.axes(frameon=False)
        show_figure = True
    else:
        show_figure = False

    # Convert to integer index in image coords.
    idx = get_idx(size, view, centre=centre, centre_other=centre_other, idx=idx, idx_mm=idx_mm, dose_data=dose_data, landmarks_data=landmarks_data, landmarks_data_other=landmarks_data_other, regions_data=regions_data, spacing=spacing, origin=origin)

    # Convert crop to Box2D.
    crop_vox_xy = None
    if crop is not None:
        if isinstance(crop, str):
            if landmarks_data is not None and crop in list(landmarks_data['landmark-id']):
                lm_data = landmarks_data[landmarks_data['landmark-id'] == crop].iloc[0]
                crop_vox_xy = __get_landmark_crop(lm_data, crop_margin_mm, size, spacing, origin, view)
            elif regions_data is not None and crop in regions_data:
                crop_vox_xy = __get_region_crop(regions_data[crop], crop_margin_mm, spacing, origin, view)
        elif isinstance(crop, LandmarkSeries):
            crop_vox_xy = __get_landmark_crop(crop, crop_margin_mm, size, spacing, origin, view)
        elif isinstance(crop, RegionArray):
            crop_vox_xy = __get_region_crop(crop, crop_margin_mm, spacing, origin, view)
        else:
            crop_vox_xy = tuple(*zip(crop))    # API accepts ((xmin, xmax), (ymin, ymax)) - convert to Box2D.
            crop_vox_xy = replace_box_none(crop_vox_xy, size, use_patient_coords=False)
    else:
        crop_vox_xy = ((0, 0), get_view_xy(size, view))  # Default to full image size.

    # Only apply aspect ratio if no transforms are being presented otherwise
    # we might end up with skewed images.
    if not aspect:
        if transform:
            aspect = 1
        else:
            aspect = get_aspect(view, spacing) 

    # Plot CT data.
    if ct_data is not None and show_ct:
        # Plot CT slice.
        ct_slice, _ = get_view_slice(ct_data, idx, view)
        ct_slice = crop_fn(ct_slice, transpose_box(crop_vox_xy), use_patient_coords=False)
        vmin, vmax = get_window(window=window, data=ct_data)
        ax.imshow(ct_slice, cmap='gray', aspect=aspect, interpolation='none', origin=get_view_origin(view)[1], vmin=vmin, vmax=vmax)

        # Highlight regions outside the window mask.
        if window_mask is not None:
            cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), 'red'))
            hw_slice = np.zeros_like(ct_slice)
            if window_mask[0] is not None:
                hw_slice[ct_slice < window_mask[0]] = 1
            if window_mask[1] is not None:
                hw_slice[ct_slice >= window_mask[1]] = 1
            ax.imshow(hw_slice, alpha=1.0, aspect=aspect, cmap=cmap, interpolation='none', origin=get_view_origin(view)[1])
    else:
        # Plot black background.
        empty_slice = np.zeros(get_view_xy(size, view))
        empty_slice = crop_fn(empty_slice, transpose_box(crop_vox_xy), use_patient_coords=False)
        ax.imshow(empty_slice, cmap='gray', aspect=aspect, interpolation='none', origin=get_view_origin(view)[1])

    # Plot crosshairs.
    if crosshairs is not None:
        # Convert crosshairs to Pixel.
        if isinstance(crosshairs, LandmarkID):
            if crosshairs not in list(landmarks_data['landmark-id']):
                raise ValueError(f"Landmark '{crosshairs}' not found in 'landmarks_data'.")
            lm_data = landmarks_data[landmarks_data['landmark-id'] == crosshairs]
            lm_data = landmarks_to_image_coords(lm_data, spacing, origin).iloc[0]
            crosshairs = get_view_xy(lm_data[list(range(3))], view)
        elif use_patient_coords and isinstance(crosshairs, Point2D):
            # Passed crosshairs should be in same coordinates as image axes. Convert to image coords
            crosshairs = (np.array(crosshairs) - get_view_xy(origin, view)) / get_view_xy(spacing, view)

        crosshairs = np.array(crosshairs) - crop_vox_xy[0]
        ax.axvline(x=crosshairs[0], color='yellow', linewidth=linewidth, linestyle='dashed')
        ax.axhline(y=crosshairs[1], color='yellow', linewidth=linewidth, linestyle='dashed')
        ch_label = crosshairs.copy()
        ch_label = ch_label + crop_vox_xy[0]
        if use_patient_coords:
            ch_label = ch_label * get_view_xy(spacing, view) + get_view_xy(origin, view)
            ch_label = f'({ch_label[0]:.1f}, {ch_label[1]:.1f})'
        else:
            ch_label = f'({ch_label[0]}, {ch_label[1]})'
        ch_origin = 10
        ax.text(crosshairs[0] + ch_origin, crosshairs[1] - ch_origin, ch_label, fontsize=8, color='yellow')

    # Plot dose data.
    isodoses = arg_to_list(isodoses, float)
    if dose_data is not None and (show_dose or len(isodoses) > 0):
        dose_slice, _ = get_view_slice(dose_data, idx, view)
        dose_slice = crop_fn(dose_slice, transpose_box(crop_vox_xy), use_patient_coords=False)

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
            imax = ax.imshow(dose_slice, aspect=aspect, cmap=dose_cmap, origin=get_view_origin(view)[1])
            if show_dose_legend:
                cbar = plt.colorbar(imax, fraction=dose_colourbar_size, pad=dose_colourbar_pad)
                cbar.set_label(label='Dose [Gray]', size=fontsize)
                cbar.ax.tick_params(labelsize=fontsize)

        # Plot isodose lines - use colourmap with alpha=1.
        if len(isodoses) > 0:
            dose_max = dose_data.max()
            isodoses = list(sorted(isodoses))
            isodose_levels = [d * dose_max for d in isodoses]
            dose_colours = dose_cmap(np.arange(dose_cmap.N))
            isodose_cmap = mpl.colors.ListedColormap(dose_colours)
            isodose_colours = [isodose_cmap(d) for d in isodoses]
            imax = ax.contour(dose_slice, colors=isodose_colours, levels=isodose_levels)

    # Plot landmarks.
    if landmarks_data is not None:
        plot_landmarks_data(landmarks_data, ax, idx, size, spacing, origin, view, crop=crop_vox_xy, dose_data=dose_data, **kwargs)

    if landmarks_data_other is not None:
        plot_landmarks_data(landmarks_data_other, ax, idx, size, spacing, origin, view, colour='red', crop=crop_vox_xy, dose_data=dose_data, **kwargs)

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
        should_show_legend = plot_regions_data(regions_data, ax, idx, aspect, **okwargs)

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
    sizes_xy = get_view_xy(size, view)
    crop_min_xy, crop_max_xy = crop_vox_xy
    spacing_xy = get_view_xy(spacing, view)
    origin_xy = get_view_xy(origin, view)
    for a, sp, o, cm, cx in zip(axes_xs, spacing_xy, origin_xy, crop_min_xy, crop_max_xy):
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
            title = ''
            if show_title_pat and pat_id is not None:
                title = pat_id
            if show_title_study and study is not None:
                prefix = '\n' if title != '' else ''
                title_study = f'...{study[-title_width:]}' if len(study) > title_width else study
                title += f'{prefix}Study: {title_study}'
                if study_date is not None:
                    title += f' ({study_date})'
            if show_title_image_series and series is not None:
                prefix = '\n' if title != '' else ''
                title_image_series = f'...{series[-title_width:]}' if len(series) > title_width else series
                title += f'{prefix}Image series: {title_image_series}'
                if series_date is not None:
                    title += f' ({series_date})'
            if show_title_dose and dose_series is not None:
                prefix = '\n' if title != '' else ''
                title_dose_series = f'...{dose_series[-title_width:]}' if len(dose_series) > title_width else dose_series
                title += f'{prefix}Dose series: {title_dose_series}'
                if dose_series_date is not None:
                    title += f' ({dose_series_date})'
            if show_title_landmark_series and landmark_series is not None and landmarks_data is not None:
                prefix = '\n' if title != '' else ''
                title_landmark_series = f'...{landmark_series[-title_width:]}' if len(landmark_series) > title_width else landmark_series
                title += f'{prefix}Landmarks series: {title_landmark_series}'
                if series_date is not None:
                    title += f' ({series_date})'
            if show_title_region_series and region_series is not None and regions_data is not None:
                prefix = '\n' if title != '' else ''
                title_region_series = f'...{region_series[-title_width:]}' if len(region_series) > title_width else region_series
                title += f'{prefix}Regions series: {title_region_series}'
                if series_date is not None:
                    title += f' ({series_date})'
            if show_title_slice:
                prefix = '\n' if title != '' else ''
                if use_patient_coords:
                    slice_mm = spacing[view] * idx + origin[view]
                    title_idx = f'{slice_mm:.1f}mm'
                else:
                    title_idx = f'{idx}/{size[view] - 1}'
                title += f'{prefix}Slice: {title_idx} ({get_axis_name(view)})'

        # Escape text if using latex.
        if escape_latex:
            title = escape_latex_fn(title)

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

def plot_patient_histogram(
    dataset_type: Dataset,
    dataset: str,
    pat: PatientIDs = 'all',
    savepath: Optional[str] = None,
    series: SeriesIDs = 'all',
    show_progress: bool = False,
    study: StudyIDs = 'all',
    **kwargs) -> None:
    set = dataset_type(dataset)
    pat_ids = set.list_patients(pat=pat)
    n_rows = len(pat_ids)

    # Get n_cols.
    n_cols = 0
    studieses = []
    for p in pat_ids:
        pat = set.patient(p)
        study_ids = pat.list_studies(study=study)
        studieses.append(ids)
        if len(ids) > n_cols:
            n_cols = len(ids)
    
    _, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    if show_progress:
        logging.info("Plotting patient histograms...")
    for i, p in tqdm(enumerate(pat_ids), disable=not show_progress):
        pat = set.patient(p)
        row_axs = axs[i]
        ss = studieses[i]
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

# Patients are plotted in rows.
# Studies/series are flattened and plotting in columns.
# Anything that applies to a series (e.g. region) can be passed
# as a list of lists.
def plot_patient(
    dataset: Dataset,
    dataset_fns: Dict[str, Callable],
    pat: PatientIDs = 'all',
    centre: Optional[Union[LandmarkID, Literal['dose'], RegionID]] = None,
    crop: Optional[Union[str, Box2D]] = None,
    dose_series: Optional[SeriesID] = None,
    isodoses: Union[float, List[float]] = [],
    landmark: Optional[LandmarkIDs] = None,
    landmark_series: Union[SeriesID, List[Optional[SeriesID]]] = None,
    landmark_other_series: Optional[SeriesID] = None,
    loadpath: Union[str, List[str]] = [],
    modality: Optional[Union[DicomModality, NiftiModality]] = None,    # Can be used instead of 'serieses'.
    region: Optional[Union[RegionIDs, List[RegionIDs]]] = None,
    region_labels: Dict[str, str] = {},
    region_series: Optional[SeriesID] = None,
    savepath: Optional[FilePath] = None,
    series: Optional[SeriesIDs] = None,
    show_dose: bool = False,
    show_progress: bool = False,
    study: Optional[StudyIDs] = None,
    **kwargs) -> None:
    isodoses = arg_to_list(isodoses, float)
    loadpaths = arg_to_list(loadpath, FilePath)
    if len(loadpaths) > 0:
        plot_saved(loadpaths)
        return
    if savepath is not None and savepath.endswith('.pdf'):
        raise ValueError(f"Savepath should not be '.pdf', issues loading without 'poppler' installed.")

    # Get patient IDs.
    arg_pat_ids = pat
    pat_ids = dataset.list_patients(pat=pat)
    if len(pat_ids) == 0:
        raise ValueError(f"No patients found for dataset '{dataset}' with IDs '{arg_pat_ids}'.")

    # Load all patient data.
    # This is the row-level data for 'plot_patients_matrix'.
    loaded_studies = []
    loaded_study_dates = []
    loaded_serieses = []
    loaded_dose_serieses = []
    loaded_dose_series_dates = []
    loaded_landmark_serieses = []
    loaded_region_serieses = []
    ct_datas = []
    dose_datas = []
    spacings = []
    origins = []
    regions_datas = []
    landmarks_datas = []
    landmarks_datas_other = []
    centre_datas = []
    crop_datas = []
    if show_progress:
        logging.info("Loading patient data...")

    # Patients are plotted one per row.
    # Studies and series are plotted within the row and grouped by study.
    for p in tqdm(pat_ids, disable=not show_progress):
        # Get studies for this patient.
        pat = dataset.patient(p)
        if study is not None:
            pat_studies = arg_to_list(study, StudyID)
        else:
            pat_studies = [pat.default_study.id]

        # Load the row data.
        row_serieses = []
        row_studies = []
        row_study_dates = []
        row_ct_datas = []
        row_dose_datas = []
        row_dose_serieses = []
        row_dose_series_dates = []
        row_landmark_serieses = []
        row_region_serieses = []
        row_spacings = []
        row_origins = []
        row_regions_datas = []
        row_landmarks_datas = []
        row_landmarks_datas_other = []
        row_centre_datas = []
        row_crop_datas = []
        for i, s in enumerate(pat_studies):
            pat_study = pat.study(s)

            # Load all image series for this study.
            # We do this here because we may not know the series ID before hand, it might be something general
            # like 'mr' to load the MR series.
            if series is not None:
                # Replace CT/MR with default series IDs.
                serieses = arg_to_list(series, (SeriesID, str))
                study_serieses = flatten_list([pat_study.list_series(i) if i in ['ct', 'mr'] else i for i in serieses])
            elif modality is not None:
                study_serieses = pat_study.list_series(modality)
            else:
                study_serieses = [pat_study.default_ct.id if pat_study.default_ct is not None else pat_study.default_mr.id]

            # Dose series can be referrring to multiple studies or series or both.
            if len(pat_studies) == 1:   # Single study, multiple image series.
                study_dose_serieses = arg_to_list(dose_series, (None, SeriesID), broadcast=len(study_serieses))
                assert len(study_dose_serieses) == len(study_serieses)
            elif len(study_serieses) == 1:  # Multiple studies, single image series.
                study_dose_serieses = arg_to_list(dose_series, (None, SeriesID), broadcast=len(pat_studies))
                assert len(study_dose_serieses) == len(pat_studies)
                study_dose_serieses = [study_dose_serieses[i]]  # Select series for current study.
            else:  # Multiple studies, multiple image series.
                study_dose_serieses = arg_to_list(dose_series, (None, SeriesID), broadcast=len(pat_studies) * len(study_serieses))
                assert len(study_dose_serieses) == len(pat_studies) * len(study_serieses)
                study_dose_serieses = study_dose_serieses[i * len(study_serieses):(i + 1) * len(study_serieses)]

            # Landmark series can be referrring to multiple studies or series or both.
            if len(pat_studies) == 1:   # Single study, multiple image series.
                study_landmark_serieses = arg_to_list(landmark_series, (None, SeriesID), broadcast=len(study_serieses))
                assert len(study_landmark_serieses) == len(study_serieses)
            elif len(study_serieses) == 1:  # Multiple studies, single image series.
                study_landmark_serieses = arg_to_list(landmark_series, (None, SeriesID), broadcast=len(pat_studies))
                assert len(study_landmark_serieses) == len(pat_studies)
                study_landmark_serieses = [study_landmark_serieses[i]]  # Select series for current study.
            else:  # Multiple studies, multiple image series.
                study_landmark_serieses = arg_to_list(landmark_series, (None, SeriesID), broadcast=len(pat_studies) * len(study_serieses))
                assert len(study_landmark_serieses) == len(pat_studies) * len(study_serieses)
                study_landmark_serieses = study_landmark_serieses[i * len(study_serieses):(i + 1) * len(study_serieses)]

            # Regions series can be referrring to multiple studies or series or both.
            if len(pat_studies) == 1:   # Single study, multiple image series.
                study_region_serieses = arg_to_list(region_series, (None, SeriesID), broadcast=len(study_serieses))
                assert len(study_region_serieses) == len(study_serieses)
            elif len(study_serieses) == 1:  # Multiple studies, single image series.
                study_region_serieses = arg_to_list(region_series, (None, SeriesID), broadcast=len(pat_studies))
                assert len(study_region_serieses) == len(pat_studies)
                study_region_serieses = [study_region_serieses[i]]  # Select series for current study.
            else:  # Multiple studies, multiple image series.
                study_region_serieses = arg_to_list(region_series, (None, SeriesID), broadcast=len(pat_studies) * len(study_serieses))
                assert len(study_region_serieses) == len(pat_studies) * len(study_serieses)
                study_region_serieses = study_region_serieses[i * len(study_serieses):(i + 1) * len(study_serieses)]

            # Regions can be referrring to multiple studies or series or both.
            if len(pat_studies) == 1:   # Single study, multiple image series.
                study_regions = arg_to_list(region, (None, RegionID), broadcast=len(study_serieses))
                assert len(study_regions) == len(study_serieses)
            elif len(study_serieses) == 1:  # Multiple studies, single image series.
                study_regions = arg_to_list(region, (None, RegionID), broadcast=len(pat_studies))
                assert len(study_regions) == len(pat_studies)
                study_regions = [study_regions[i]]  # Select series for current study.
            else:  # Multiple studies, multiple image series.
                study_regions = arg_to_list(region, (None, RegionID), broadcast=len(pat_studies) * len(study_serieses))
                assert len(study_regions) == len(pat_studies) * len(study_serieses)
                study_regions = study_regions[i * len(study_serieses):(i + 1) * len(study_serieses)]

            # Add data for each series.
            for ss, sd, sl, sr, srr in zip(study_serieses, study_dose_serieses, study_landmark_serieses, study_region_serieses, study_regions):
                row_serieses.append(ss)
                row_landmark_serieses.append(sl)
                row_studies.append(s)
                row_study_dates.append(dataset_fns['study_datetime'](study))

                # Load image data (e.g. CT/MR).
                ct_series = dataset_fns['ct_series'](pat_study, ss)
                row_ct_datas.append(ct_series.data)
                row_spacings.append(ct_series.spacing)
                row_origins.append(ct_series.origin)

                # Get landmarks data.
                if sl is not None:
                    lm_series = dataset_fns['landmark_series'](pat_study, sl)
                else:
                    lm_series = dataset_fns['default_landmarks'](pat_study)
                other_lm_series = pat_study.series(landmark_other_series, 'rtstruct') if landmark_other_series is not None else None
                if lm_series is not None:
                    lm_data = dataset_fns['landmarks_data'](lm_series, landmark) if landmark is not None else None
                else:
                    lm_data = None
                if other_lm_series is not None:
                    other_lm_data = dataset_fns['landmarks_data'](other_lm_series, landmark) if landmark is not None else None
                else:
                    other_lm_data = None
                row_landmarks_datas.append(lm_data)
                row_landmarks_datas_other.append(other_lm_data)

                # Get regions data.
                if sr is not None:
                    r_series = dataset_fns['region_series'](pat_study, sr)
                else:
                    r_series = dataset_fns['default_regions'](pat_study)
                row_region_serieses.append(r_series.id if r_series is not None else None)
                rdata = dataset_fns['regions_data'](r_series, srr) if r_series is not None and srr is not None else None
                row_regions_datas.append(rdata)

                # Load dose data from the same study - and resample to image spacing.
                if (show_dose or len(isodoses) > 0) and dataset_fns['has_dose'](pat_study):
                    if sd is not None:
                        dose_s = dataset_fns['dose_series'](pat_study, sd)
                        resample_kwargs = dict(
                            origin=dose_s.origin,
                            output_origin=ct_series.origin,
                            output_size=ct_series.size,
                            output_spacing=ct_series.spacing,
                            spacing=dose_s.spacing,
                        )
                        dose_data = resample(dose_s.data, **resample_kwargs)
                        row_dose_datas.append(dose_data)
                        row_dose_serieses.append(dose_s.id)
                        row_dose_series_dates.append(dose_s.date)
                    else:
                        row_dose_datas.append(None)
                        row_dose_serieses.append(None)
                        row_dose_series_dates.append(None)
                else:
                    row_dose_datas.append(None)
                    row_dose_serieses.append(None)
                    row_dose_series_dates.append(None)

                # Handle centre.
                c = None
                if centre is not None:
                    if centre == 'dose':
                        c = centre
                    elif isinstance(centre, (LandmarkID, RegionID)):
                        # If data isn't in landmarks/regions_data then pass the data as 'centre', otherwise 'centre' can reference 
                        # the data in 'landmarks/regions_data'.
                        if pat_study.has_landmark(centre):
                            c = pat_study.landmarks_data(landmark=centre).iloc[0] if lm_data is None or centre not in list(lm_data['landmark-id']) else centre
                        elif pat_study.has_region(centre):
                            c = pat_study.regions_data(region=centre)[centre] if rdata is None or centre not in rdata else centre
                        else:
                            raise ValueError(f"Study {pat_study} has no landmark/regions with ID '{centre}' for 'centre'.")
                row_centre_datas.append(c)

                # Add crop - load data if necessary.
                c = None
                if crop is not None:
                    if isinstance(crop, str):
                        if pat_study.has_region(crop):
                            if rdata is not None and crop in rdata:
                                c = crop    # Pass string, it will be read from 'regions_data' by 'plot_patient'.
                            else:
                                c = pat_study.regions_data(region=crop)[crop]  # Load RegionLabel.
                        elif pat_study.has_landmark(crop):
                            if lm_data is not None and crop in list(lm_data['landmark-id']):
                                c = crop
                            else:
                                c = pat_study.landmarks_data(landmark=crop).iloc[0]    # Load LandmarkSeries.
                        else:
                            raise ValueError(f"Study '{pat_study}' has no landmark/region ID '{crop}' for crop.")
                row_crop_datas.append(c)

        # Apply region labels.
        # This should maybe be moved to base 'plot_patient'? All of the dataset-specific plotting functions
        # use this. Of course 'plot_patient' API would change to include 'region_labels' as an argument.
        for i in range(len(row_serieses)):
            row_regions_datas[i], row_centre_datas[i], row_crop_datas[i] = __map_regions(region_labels, row_regions_datas[i], row_centre_datas[i], row_crop_datas[i])

        loaded_studies.append(row_studies)
        loaded_study_dates.append(row_study_dates)
        loaded_serieses.append(row_serieses)
        loaded_dose_serieses.append(row_dose_serieses)
        loaded_dose_series_dates.append(row_dose_series_dates)
        loaded_landmark_serieses.append(row_landmark_serieses)
        loaded_region_serieses.append(row_region_serieses)
        ct_datas.append(row_ct_datas)
        dose_datas.append(row_dose_datas)
        spacings.append(row_spacings)
        origins.append(row_origins)
        regions_datas.append(row_regions_datas)
        landmarks_datas.append(row_landmarks_datas)
        landmarks_datas_other.append(row_landmarks_datas_other)
        centre_datas.append(row_centre_datas)
        crop_datas.append(row_crop_datas)

    # Plot.
    okwargs = dict(
        centres=centre_datas,
        crops=crop_datas,
        ct_datas=ct_datas,
        dose_datas=dose_datas,
        dose_serieses=loaded_dose_serieses,
        dose_series_dates=loaded_dose_series_dates,
        isodoses=isodoses,
        landmarks_datas=landmarks_datas,
        landmarks_datas_other=landmarks_datas_other,
        landmark_serieses=loaded_landmark_serieses,
        origins=origins,
        regions_datas=regions_datas,
        region_serieses=loaded_region_serieses,
        savepath=savepath,
        serieses=loaded_serieses,
        show_progress=show_progress,
        show_dose=show_dose,
        spacings=spacings,
        study_dates=loaded_study_dates,
        study=loaded_studies,
    )
    plot_patients_matrix(pat_ids, **okwargs, **kwargs)

@alias_kwargs(
    ('ti', 'transpose_images')
)
def plot_patients_matrix(
    # Allows us to plot multiple patients (rows) and patient studies, series, and views (columns).
    pat_ids: Union[str, List[str]],
    ax: Optional[mpl.axes.Axes] = None,
    centres: Optional[Union[LandmarkSeries, LandmarkID, Literal['dose'], RegionArray, RegionID, List[Union[LandmarkSeries, LandmarkID, Literal['dose'], RegionArray, RegionID]], List[Union[LandmarkSeries, LandmarkID, RegionArray, RegionID, List[Union[LandmarkSeries, LandmarkID, RegionArray, RegionID]]]]]] = None,
    crops: Optional[Union[str, np.ndarray, Box2D, List[Union[str, np.ndarray, Box2D]]]] = None,
    ct_datas: Optional[Union[CtImageArray, List[CtImageArray], List[Union[CtImageArray, List[CtImageArray]]]]] = None,
    dose_datas: Optional[Union[DoseImageArray, List[DoseImageArray], List[Union[DoseImageArray, List[DoseImageArray]]]]] = None,
    dose_serieses: Union[SeriesIDs, List[SeriesIDs]] = None,
    dose_series_dates: Union[str, List[str]] = None,
    figsize: Tuple[int, int] = (46, 12),    # In cm.
    landmarks_datas: Optional[Union[LandmarksFrame, List[LandmarksFrame], List[Union[LandmarksFrame, List[LandmarksFrame]]]]] = None,
    landmarks_datas_other: Optional[Union[LandmarksFrame, List[LandmarksFrame], List[Union[LandmarksFrame, List[LandmarksFrame]]]]] = None,
    landmark_serieses: Union[SeriesIDs, List[SeriesIDs]] = None,
    origins: Union[Point3D, List[Point3D], List[Union[Point3D, List[Point3D]]]] = None,
    regions_datas: Optional[Union[RegionArrays, List[RegionArrays], List[Union[RegionArrays, List[RegionArrays]]]]] = None,
    region_serieses: Union[SeriesIDs, List[SeriesIDs]] = None,
    savepath: Optional[str] = None,
    serieses: Union[SeriesIDs, List[SeriesIDs]] = None,
    show_progress: bool = False,
    spacings: Union[Spacing3D, List[Spacing3D], List[Union[Spacing3D, List[Spacing3D]]]] = None,
    study_dates: Union[str, List[str]] = None,
    study: Union[StudyIDs, List[StudyIDs]] = None,
    transpose_images: bool = False,
    view: Union[Axis, List[Axis], Literal['all']] = 0,
    **kwargs) -> None:
    # Handle args.
    pat_ids = arg_to_list(pat_ids, PatientID)
    n_rows = len(pat_ids)
    studies = arg_to_list(study, StudyID, broadcast=n_rows)
    study_dates = arg_to_list(study_dates, str, broadcast=n_rows)
    serieses = arg_to_list(serieses, SeriesID, broadcast=n_rows)
    spacings = arg_to_list(spacings, Spacing3D, broadcast=n_rows)
    centres = arg_to_list(centres, (None, LandmarkSeries, LandmarkID, RegionArray, RegionID), broadcast=n_rows)
    crops = arg_to_list(crops, (None, LandmarkID, Box2D, RegionID), broadcast=n_rows)
    ct_datas = arg_to_list(ct_datas, (None, CtImageArray), broadcast=n_rows)
    dose_datas = arg_to_list(dose_datas, (None, DoseImageArray), broadcast=n_rows)
    dose_series_dates = arg_to_list(dose_series_dates, str, broadcast=n_rows)
    dose_serieses = arg_to_list(dose_serieses, SeriesID, broadcast=n_rows)
    landmarks_datas = arg_to_list(landmarks_datas, (None, LandmarksFrame), broadcast=n_rows)
    landmarks_datas_other = arg_to_list(landmarks_datas_other, (None, LandmarksFrame), broadcast=n_rows)
    landmark_serieses = arg_to_list(landmark_serieses, SeriesID, broadcast=n_rows)
    regions_datas = arg_to_list(regions_datas, (None, RegionArrays), broadcast=n_rows)
    region_serieses = arg_to_list(region_serieses, SeriesID, broadcast=n_rows)
    views = arg_to_list(view, int, literals={ 'all': tuple(range(3)) })
    n_series_max = np.max([len(ss) for ss in serieses])
    n_cols = len(views) * n_series_max
    if savepath is not None and savepath.endswith('.pdf'):
        raise ValueError(f"Use '.jpg' or '.png' for 'savepath', '.pdf' is tricky to load without 'poppler' installed.")

    # Convert figsize from cm to inches.
    figsize = figsize_to_inches(figsize)

    # Create axes.
    if ax is None:
        if n_rows > 1 or n_cols > 1:
            # Subplots for multiple views.
            n_plots = (n_rows, n_cols) if not transpose_images else (n_cols, n_rows)
            figsize = (figsize[0], n_rows * figsize[1]) if not transpose_images else (figsize[0], n_cols * figsize[1])
            _, axs = plt.subplots(*n_plots, figsize=figsize, gridspec_kw={ 'hspace': 0.5 }, squeeze=False)
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
            seriesx = j % n_series_max
            view_idx = j // n_series_max
            ct_data = ct_datas[i][seriesx] if isinstance(ct_datas[i], list) else ct_datas[i]
            dose_data = dose_datas[i][seriesx] if isinstance(dose_datas[i], list) else dose_datas[i]
            dose_series = dose_serieses[i][seriesx] if isinstance(dose_serieses[i], list) else dose_serieses[i]
            dose_series_date = dose_series_dates[i][seriesx] if isinstance(dose_series_dates[i], list) else dose_series_dates[i]
            spacing = spacings[i][seriesx] if isinstance(spacings[i], list) else spacings[i]
            origin = origins[i][seriesx] if isinstance(origins[i], list) else origins[i]
            regions_data = regions_datas[i][seriesx] if isinstance(regions_datas[i], list) else regions_datas[i]
            landmarks_data = landmarks_datas[i][seriesx] if isinstance(landmarks_datas[i], list) else landmarks_datas[i]
            landmarks_data_other = landmarks_datas_other[i][seriesx] if isinstance(landmarks_datas_other[i], list) else landmarks_datas_other[i]
            landmark_series = landmark_serieses[i][seriesx] if isinstance(landmark_serieses[i], list) else landmark_serieses[i]
            crop = crops[i][seriesx] if isinstance(crops[i], list) else crops[i]
            centre = centres[i][seriesx] if isinstance(centres[i], list) else centres[i]
            region_series = region_serieses[i][seriesx] if isinstance(region_serieses[i], list) else region_serieses[i]
            series = serieses[i][seriesx] if isinstance(serieses[i], list) else serieses[i]
            study = studies[i][seriesx] if isinstance(studies[i], list) else studies[i]
            study_date = study_dates[i][seriesx] if isinstance(study_dates[i], list) else study_dates[i]
            view = views[view_idx] if len(views) > 1 else views[0]

            plot_single_patient(
                ct_data.shape,
                spacing,
                ax=axs[i][j] if not transpose_images else axs[j][i],
                centre=centre,
                close_figure=False,
                crop=crop,
                ct_data=ct_data,
                dose_data=dose_data,
                dose_series=dose_series,
                dose_series_date=dose_series_date,
                landmarks_data=landmarks_data,
                landmarks_data_other=landmarks_data_other,
                landmark_series=landmark_series,
                origin=origin,
                pat_id=pat_id,
                regions_data=regions_data,
                region_series=region_series,
                series=series,
                study_date=study_date,
                study=study,
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

def __get_landmark_crop(
    landmark: LandmarkSeries,
    margin_mm: float,
    size: Size3D,
    spacing: Spacing3D,
    origin: Point3D,
    view: Axis) -> Box2D:
    # Add crop margin.
    landmark = landmark[range(3)].to_numpy()
    min_mm = landmark - margin_mm
    max_mm = landmark + margin_mm

    # Convert to image coords.
    min_vox = tuple(np.floor((min_mm - origin) / spacing).astype(int))
    max_vox = tuple(np.ceil((max_mm - origin) / spacing).astype(int))

    # Don't pad original image.
    min_vox = tuple(int(v) for v in np.clip(min_vox, a_min=0, a_max=None))
    max_vox = tuple(int(v) for v in np.clip(max_vox, a_min=None, a_max=size))

    # Select 2D component.
    min_vox = get_view_xy(min_vox, view)
    max_vox = get_view_xy(max_vox, view)

    return min_vox, max_vox

def __map_regions(
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

def __get_region_crop(
    data: RegionArray,
    margin_mm: float,
    spacing: Spacing3D,
    origin: Point3D,
    view: Axis) -> Box2D:
    # Get region extent.
    fov_min_mm, fov_max_mm = foreground_fov(data, spacing=spacing, origin=origin)

    # Add crop margin.
    min_mm = np.array(fov_min_mm) - margin_mm
    max_mm = np.array(fov_max_mm) + margin_mm

    # Convert to image coords.
    min_vox = tuple(np.floor((min_mm - origin) / spacing).astype(int))
    max_vox = tuple(np.ceil((max_mm - origin) / spacing).astype(int))

    # Don't pad original image.
    min_vox = tuple(int(v) for v in np.clip(min_vox, a_min=0, a_max=None))
    max_vox = tuple(int(v) for v in np.clip(max_vox, a_min=None, a_max=data.shape))     # Should be inclusive to maintain symmetry.

    # Select 2D component.
    min_vox = get_view_xy(min_vox, view)
    max_vox = get_view_xy(max_vox, view)

    return min_vox, max_vox

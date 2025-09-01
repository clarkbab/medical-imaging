import matplotlib as mpl
import torchio
from typing import *

from mymi.constants import *
from mymi.typing import *
from mymi.utils import *

from .data import plot_histograms
from .plotting import *

@alias_kwargs(('upc', 'use_patient_coords'))
def plot_patient(
    size: Size3D,
    spacing: Spacing3D,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    ax: Optional[mpl.axes.Axes] = None,
    centre: Optional[Union[LandmarkSeries, LandmarkID, Literal['dose'], RegionArray, RegionID]] = None,
    centre_other: bool = False,
    colours: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[LandmarkSeries, LandmarkID, Box2D, RegionArray, RegionID]] = None,    # Uses 'region_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    crop_margin_mm: float = 100,                                       # Applied if cropping to 'region_data' or 'np.ndarray'.
    crosshairs: Optional[Union[Pixel, Point2D, LandmarkSeries]] = None,
    ct_data: Optional[CtImageArray] = None,
    dose_alpha_min: float = 0.3,
    dose_alpha_max: float = 1.0,
    dose_cmap: str = 'turbo',
    dose_cmap_trunc: float = 0.15,
    dose_colourbar_pad: float = 0.05,
    dose_colourbar_size: float = 0.03,
    dose_data: Optional[DoseImageArray] = None,
    dose_series_date: Optional[str] = None,
    dose_series_id: Optional[SeriesID] = None,
    escape_latex: bool = False,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'region_data' if 'str', else 'np.ndarray') and min/max of extent.
    figsize: Tuple[float, float] = (36, 12),
    fontsize: int = 12,
    idx: Optional[float] = None,
    idx_mm: Optional[float] = None,
    isodoses: Union[float, List[float]] = [],
    landmark_data: Optional[LandmarksFrame] = None,     # All landmarks are plotted.
    landmark_data_other: Optional[LandmarksFrame] = None,     # Plotted as 'red' landmarks, e.g. from registration.
    legend_bbox_to_anchor: Optional[Tuple[float, float]] = (1, 1),
    legend_loc: Union[str, Tuple[float, float]] = 'upper left',
    legend_show_all_regions: bool = False,
    linewidth: float = 0.5,
    linewidth_legend: float = 8,
    norm: Optional[Tuple[float, float]] = None,
    offset: Optional[Point3D] = None,
    pat_id: Optional[PatientID] = None,
    postproc: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    region_data: Optional[RegionArrays] = None,         # All regions are plotted.
    savepath: Optional[str] = None,
    series_date: Optional[str] = None,
    series_id: Optional[SeriesID] = None,
    show_axes: bool = True,
    show_ct: bool = True,
    show_dose: bool = False,
    show_dose_legend: bool = True,
    show_extent: bool = False,
    show_legend: bool = False,
    show_title: bool = True,
    show_title_dose: bool = False,
    show_title_pat: bool = True,
    show_title_series: bool = False,
    show_title_slice: bool = True,
    show_title_study: bool = True,
    show_x_label: bool = True,
    show_x_ticks: bool = True,
    show_y_label: bool = True,
    show_y_ticks: bool = True,
    study_date: Optional[str] = None,
    study_id: Optional[StudyID] = None,
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
    idx = get_idx(size, view, centre=centre, centre_other=centre_other, idx=idx, idx_mm=idx_mm, dose_data=dose_data, landmark_data=landmark_data, landmark_data_other=landmark_data_other, region_data=region_data, spacing=spacing, offset=offset)

    # Convert crop to Box2D.
    crop_vox_xy = None
    if crop is not None:
        if isinstance(crop, str):
            if landmark_data is not None and crop in list(landmark_data['landmark-id']):
                lm_data = landmark_data[landmark_data['landmark-id'] == crop].iloc[0]
                crop_vox_xy = __get_landmark_crop(lm_data, crop_margin_mm, size, spacing, offset, view)
            elif region_data is not None and crop in region_data:
                crop_vox_xy = __get_region_crop(region_data[crop], crop_margin_mm, spacing, offset, view)
        elif isinstance(crop, LandmarkSeries):
            crop_vox_xy = __get_landmark_crop(crop, crop_margin_mm, size, spacing, offset, view)
        elif isinstance(centre, RegionArray):
            crop_vox_xy = __get_region_crop(crop, crop_margin_mm, spacing, offset, view)
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
        ax.imshow(ct_slice, cmap='gray', aspect=aspect, interpolation='none', origin=get_origin(view), vmin=vmin, vmax=vmax)

        # Highlight regions outside the window mask.
        if window_mask is not None:
            cmap = mpl.colors.ListedColormap(((1, 1, 1, 0), 'red'))
            hw_slice = np.zeros_like(ct_slice)
            if window_mask[0] is not None:
                hw_slice[ct_slice < window_mask[0]] = 1
            if window_mask[1] is not None:
                hw_slice[ct_slice >= window_mask[1]] = 1
            ax.imshow(hw_slice, alpha=1.0, aspect=aspect, cmap=cmap, interpolation='none', origin=get_origin(view))
    else:
        # Plot black background.
        empty_slice = np.zeros(get_view_xy(size, view))
        empty_slice = crop_fn(empty_slice, transpose_box(crop_vox_xy), use_patient_coords=False)
        ax.imshow(empty_slice, cmap='gray', aspect=aspect, interpolation='none', origin=get_origin(view))

    # Plot crosshairs.
    if crosshairs is not None:
        # Convert crosshairs to Pixel.
        if isinstance(crosshairs, LandmarkID):
            if crosshairs not in list(landmark_data['landmark-id']):
                raise ValueError(f"Landmark '{crosshairs}' not found in 'landmark_data'.")
            lm_data = landmark_data[landmark_data['landmark-id'] == crosshairs]
            lm_data = landmarks_to_image_coords(lm_data, spacing, offset).iloc[0]
            crosshairs = get_view_xy(lm_data[list(range(3))], view)
        elif use_patient_coords and isinstance(crosshairs, Point2D):
            # Passed crosshairs should be in same coordinates as image axes. Convert to image coords
            crosshairs = (np.array(crosshairs) - get_view_xy(offset, view)) / get_view_xy(spacing, view)

        crosshairs = np.array(crosshairs) - crop_vox_xy[0]
        ax.axvline(x=crosshairs[0], color='yellow', linewidth=linewidth, linestyle='dashed')
        ax.axhline(y=crosshairs[1], color='yellow', linewidth=linewidth, linestyle='dashed')
        ch_label = crosshairs.copy()
        ch_label = ch_label + crop_vox_xy[0]
        if use_patient_coords:
            ch_label = ch_label * get_view_xy(spacing, view) + get_view_xy(offset, view)
            ch_label = f'({ch_label[0]:.1f}, {ch_label[1]:.1f})'
        else:
            ch_label = f'({ch_label[0]}, {ch_label[1]})'
        ch_offset = 10
        ax.text(crosshairs[0] + ch_offset, crosshairs[1] - ch_offset, ch_label, fontsize=8, color='yellow')

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
            imax = ax.imshow(dose_slice, aspect=aspect, cmap=dose_cmap, origin=get_origin(view))
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
    if landmark_data is not None:
        plot_landmark_data(landmark_data, ax, idx, size, spacing, offset, view, crop=crop_vox_xy, dose_data=dose_data, **kwargs)

    if landmark_data_other is not None:
        plot_landmark_data(landmark_data_other, ax, idx, size, spacing, offset, view, colour='red', crop=crop_vox_xy, dose_data=dose_data, **kwargs)

    if region_data is not None:
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
        should_show_legend = plot_region_data(region_data, ax, idx, aspect, **okwargs)

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
    offset_xy = get_view_xy(offset, view)
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
            title = ''
            if show_title_pat and pat_id is not None:
                title = pat_id
            if show_title_study and study_id is not None:
                prefix = '\n' if title != '' else ''
                title_study_id = f'...{study_id[-title_width:]}' if len(study_id) > title_width else study_id
                title += f'{prefix}Study: {title_study_id}'
                if study_date is not None:
                    title += f' ({study_date})'
            if show_title_series and series_id is not None:
                prefix = '\n' if title != '' else ''
                title_series_id = f'...{series_id[-title_width:]}' if len(series_id) > title_width else series_id
                title += f'{prefix}Series: {title_series_id}'
                if series_date is not None:
                    title += f' ({series_date})'
            if show_title_slice:
                prefix = '\n' if title != '' else ''
                if use_patient_coords:
                    slice_mm = spacing[view] * idx + offset[view]
                    title_idx = f'{slice_mm:.1f}mm'
                else:
                    title_idx = f'{idx}/{size[view] - 1}'
                title += f'{prefix}Slice: {title_idx} ({get_axis_name(view)})'
            if show_title_dose and dose_series_id is not None:
                prefix = '\n' if title != '' else ''
                title_dose_series_id = f'...{dose_series_id[-title_width:]}' if len(dose_series_id) > title_width else dose_series_id
                title += f'{prefix}Dose series: {title_dose_series_id}'
                if dose_series_date is not None:
                    title += f' ({dose_series_date})'

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

def plot_patients(
    dataset: Dataset,
    dataset_fns: Dict[str, Callable],
    pat_ids: PatientIDs = 'all',
    centre: Optional[Union[LandmarkID, Literal['dose'], RegionID]] = None,
    crop: Optional[Union[str, Box2D]] = None,
    dose_series_id: Optional[SeriesID] = None,
    isodoses: Union[float, List[float]] = [],
    landmark_ids: Optional[LandmarkIDs] = None,
    landmarks_series_id: Optional[SeriesID] = None,
    landmarks_other_series_id: Optional[SeriesID] = None,
    loadpaths: Union[str, List[str]] = [],
    modality: Optional[Union[DicomModality, NiftiModality]] = None,    # Can be used instead of 'series_ids'.
    region_ids: Optional[RegionIDs] = None,
    region_labels: Dict[str, str] = {},
    regions_series_id: Optional[SeriesID] = None,
    savepath: Optional[FilePath] = None,
    series_ids: Optional[Union[SeriesID, List[SeriesID], Literal['all']]] = None,
    show_dose: bool = False,
    show_progress: bool = False,
    study_ids: Optional[Union[StudyID, List[StudyID], Literal['all']]] = None,
    **kwargs) -> None:
    isodoses = arg_to_list(isodoses, float)
    if len(loadpaths) > 0:
        plot_saved(loadpaths)
        return
    if savepath is not None and savepath.endswith('.pdf'):
        raise ValueError(f"Savepath should not be '.pdf', issues loading without 'poppler' installed.")

    # Get patient IDs.
    arg_pat_ids = pat_ids
    pat_ids = dataset.list_patients(pat_ids=pat_ids)
    if len(pat_ids) == 0:
        raise ValueError(f"No patients found for dataset '{dataset}' with IDs '{arg_pat_ids}'.")

    # Load all patient data.
    # This is the row-level data for 'plot_patients_matrix'.
    loaded_study_ids = []
    loaded_study_dates = []
    loaded_series_ids = []
    loaded_dose_series_ids = []
    loaded_dose_series_dates = []
    ct_datas = []
    dose_datas = []
    spacings = []
    offsets = []
    region_datas = []
    landmark_datas = []
    landmark_datas_other = []
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
        pat = dataset.patient(p)
        if study_ids is None:
            pat_study_ids = [pat.default_study.id]
        else:
            pat_study_ids = pat.list_studies(study_ids=study_ids)

        row_series_ids = []
        row_study_ids = []
        row_study_dates = []
        row_ct_datas = []
        row_dose_datas = []
        row_dose_series_ids = []
        row_dose_series_dates = []
        row_spacings = []
        row_offsets = []
        row_region_datas = []
        row_landmark_datas = []
        row_landmark_datas_other = []
        row_centre_datas = []
        row_crop_datas = []
        for s in pat_study_ids:
            study = pat.study(s)

            # Determine image series.
            if series_ids is not None:
                # Replace CT/MR with default series IDs.
                study_series_ids = [study.list_series(i) if i in ['ct', 'mr'] else i for i in series_ids].flatten()
            elif modality is not None:
                study_series_ids = study.list_series(modality)
            else:
                study_series_ids = [study.default_ct.id if study.default_ct is not None else study.default_mr.id]

            # Add data for each series.
            for ss in study_series_ids:
                row_series_ids.append(ss)
                row_study_ids.append(s)
                row_study_dates.append(dataset_fns['study_datetime'](study))

                # Load image data (e.g. CT/MR).
                ct_series = dataset_fns['ct_series'](study, ss)
                row_ct_datas.append(ct_series.data)
                row_spacings.append(ct_series.spacing)
                row_offsets.append(ct_series.offset)

                # Determine landmarks/regions series.
                if landmarks_series_id is not None:
                    landmarks_series = dataset_fns['landmarks_series'](study, landmarks_series_id)
                else:
                    landmarks_series = dataset_fns['default_landmarks'](study)
                other_landmarks_series = study.series(landmarks_other_series_id, 'rtstruct') if landmarks_other_series_id is not None else None
                if regions_series_id is not None:
                    regions_series = dataset_fns['regions_series'](study, regions_series_id)
                else:
                    regions_series = dataset_fns['default_regions'](study)

                # Load landmarks/regions data.
                if landmarks_series is not None:
                    lm_data = dataset_fns['landmark_data'](landmarks_series, landmark_ids) if landmark_ids is not None else None
                else:
                    lm_data = None
                row_landmark_datas.append(lm_data)
                if other_landmarks_series is not None:
                    other_lm_data = dataset_fns['landmark_data'](other_landmarks_series, landmark_ids) if landmark_ids is not None else None
                else:
                    other_lm_data = None
                row_landmark_datas_other.append(other_lm_data)
                rdata = dataset_fns['region_data'](regions_series, region_ids) if region_ids is not None else None
                row_region_datas.append(rdata)

                # Load dose data from the same study - and resample to image spacing.
                if (show_dose or len(isodoses) > 0) and dataset_fns['has_dose'](study):
                    if dose_series_id is not None:
                        dose_series = dataset_fns['dose_series'](study, dose_series_id)
                    else:
                        dose_series = dataset_fns['default_dose'](study)
                    resample_kwargs = dict(
                        offset=dose_series.offset,
                        output_offset=ct_series.offset,
                        output_size=ct_series.size,
                        output_spacing=ct_series.spacing,
                        spacing=dose_series.spacing,
                    )
                    dose_data = resample(dose_series.data, **resample_kwargs)
                    row_dose_datas.append(dose_data)
                    row_dose_series_ids.append(dose_series.id)
                    row_dose_series_dates.append(dose_series.date)
                else:
                    row_dose_datas.append(None)
                    row_dose_series_ids.append(None)
                    row_dose_series_dates.append(None)

                # Handle centre.
                c = None
                if centre is not None:
                    if centre == 'dose':
                        c = centre
                    elif isinstance(centre, (LandmarkID, RegionID)):
                        # If data isn't in landmarks/region_data then pass the data as 'centre', otherwise 'centre' can reference 
                        # the data in 'landmarks/region_data'.
                        if study.has_landmark(centre):
                            c = study.landmark_data(landmark_ids=centre).iloc[0] if lm_data is None or centre not in list(lm_data['landmark-id']) else centre
                        elif study.has_region(centre):
                            c = study.region_data(region_ids=centre)[centre] if rdata is None or centre not in rdata else centre
                        else:
                            raise ValueError(f"Study {study} has no landmark/regions with ID '{centre}' for 'centre'.")
                row_centre_datas.append(c)

                # Add crop - load data if necessary.
                c = None
                if crop is not None:
                    if isinstance(crop, str):
                        if study.has_region(crop):
                            if rdata is not None and crop in rdata:
                                c = crop    # Pass string, it will be read from 'region_data' by 'plot_patient'.
                            else:
                                c = study.region_data(region_ids=crop)[crop]  # Load RegionLabel.
                        elif study.has_landmark(crop):
                            if lm_data is not None and crop in list(lm_data['landmark-id']):
                                c = crop
                            else:
                                c = study.landmark_data(landmark_ids=crop).iloc[0]    # Load LandmarkSeries.
                row_crop_datas.append(c)

        # Apply region labels.
        # This should maybe be moved to base 'plot_patient'? All of the dataset-specific plotting functions
        # use this. Of course 'plot_patient' API would change to include 'region_labels' as an argument.
        for i in range(len(row_series_ids)):
            row_region_datas[i], row_centre_datas[i], row_crop_datas[i] = __map_regions(region_labels, row_region_datas[i], row_centre_datas[i], row_crop_datas[i])

        loaded_study_ids.append(row_study_ids)
        loaded_study_dates.append(row_study_dates)
        loaded_series_ids.append(row_series_ids)
        loaded_dose_series_ids.append(row_dose_series_ids)
        loaded_dose_series_dates.append(row_dose_series_dates)
        ct_datas.append(row_ct_datas)
        dose_datas.append(row_dose_datas)
        spacings.append(row_spacings)
        offsets.append(row_offsets)
        region_datas.append(row_region_datas)
        landmark_datas.append(row_landmark_datas)
        landmark_datas_other.append(row_landmark_datas_other)
        centre_datas.append(row_centre_datas)
        crop_datas.append(row_crop_datas)

    # Plot.
    okwargs = dict(
        centres=centre_datas,
        crops=crop_datas,
        ct_datas=ct_datas,
        dose_datas=dose_datas,
        dose_series_dates=loaded_dose_series_dates,
        dose_series_ids=loaded_dose_series_ids,
        isodoses=isodoses,
        landmark_datas=landmark_datas,
        landmark_datas_other=landmark_datas_other,
        offsets=offsets,
        region_datas=region_datas,
        savepath=savepath,
        series_ids=loaded_series_ids,
        show_progress=show_progress,
        show_dose=show_dose,
        spacings=spacings,
        study_dates=loaded_study_dates,
        study_ids=loaded_study_ids,
    )
    plot_patients_matrix(pat_ids, **okwargs, **kwargs)

def plot_patients_matrix(
    # Allows us to plot multiple patients (rows) and patient studies, series, and views (columns).
    pat_ids: Union[str, List[str]],
    ax: Optional[mpl.axes.Axes] = None,
    centres: Optional[Union[LandmarkSeries, LandmarkID, Literal['dose'], RegionArray, RegionID, List[Union[LandmarkSeries, LandmarkID, Literal['dose'], RegionArray, RegionID]], List[Union[LandmarkSeries, LandmarkID, RegionArray, RegionID, List[Union[LandmarkSeries, LandmarkID, RegionArray, RegionID]]]]]] = None,
    crops: Optional[Union[str, np.ndarray, Box2D, List[Union[str, np.ndarray, Box2D]]]] = None,
    ct_datas: Optional[Union[CtImageArray, List[CtImageArray], List[Union[CtImageArray, List[CtImageArray]]]]] = None,
    dose_datas: Optional[Union[DoseImageArray, List[DoseImageArray], List[Union[DoseImageArray, List[DoseImageArray]]]]] = None,
    dose_series_dates: Union[str, List[str]] = None,
    dose_series_ids: Union[SeriesIDs, List[SeriesIDs]] = None,
    figsize: Tuple[int, int] = (46, 12),    # In cm.
    landmark_datas: Optional[Union[LandmarksFrame, List[LandmarksFrame], List[Union[LandmarksFrame, List[LandmarksFrame]]]]] = None,
    landmark_datas_other: Optional[Union[LandmarksFrame, List[LandmarksFrame], List[Union[LandmarksFrame, List[LandmarksFrame]]]]] = None,
    offsets: Union[Point3D, List[Point3D], List[Union[Point3D, List[Point3D]]]] = None,
    region_datas: Optional[Union[RegionArrays, List[RegionArrays], List[Union[RegionArrays, List[RegionArrays]]]]] = None,
    savepath: Optional[str] = None,
    series_ids: Union[SeriesIDs, List[SeriesIDs]] = None,
    show_progress: bool = False,
    spacings: Union[Spacing3D, List[Spacing3D], List[Union[Spacing3D, List[Spacing3D]]]] = None,
    study_dates: Union[str, List[str]] = None,
    study_ids: Union[StudyIDs, List[StudyIDs]] = None,
    views: Union[Axis, List[Axis], Literal['all']] = 0,
    **kwargs) -> None:
    # Handle args.
    pat_ids = arg_to_list(pat_ids, PatientID)
    n_rows = len(pat_ids)
    study_ids = arg_to_list(study_ids, StudyID, broadcast=n_rows)
    study_dates = arg_to_list(study_dates, str, broadcast=n_rows)
    series_ids = arg_to_list(series_ids, SeriesID, broadcast=n_rows)
    spacings = arg_to_list(spacings, Spacing3D, broadcast=n_rows)
    centres = arg_to_list(centres, (None, LandmarkSeries, LandmarkID, RegionArray, RegionID), broadcast=n_rows)
    crops = arg_to_list(crops, (None, LandmarkID, Box2D, RegionID), broadcast=n_rows)
    ct_datas = arg_to_list(ct_datas, (None, CtImageArray), broadcast=n_rows)
    dose_datas = arg_to_list(dose_datas, (None, DoseImageArray), broadcast=n_rows)
    dose_series_dates = arg_to_list(dose_series_dates, str, broadcast=n_rows)
    dose_series_ids = arg_to_list(dose_series_ids, SeriesID, broadcast=n_rows)
    landmark_datas = arg_to_list(landmark_datas, (None, LandmarksFrame), broadcast=n_rows)
    landmark_datas_other = arg_to_list(landmark_datas_other, (None, LandmarksFrame), broadcast=n_rows)
    region_datas = arg_to_list(region_datas, (None, RegionArrays), broadcast=n_rows)
    views = arg_to_list(views, int, literals={ 'all': tuple(range(3)) })
    n_series_max = np.max([len(ss) for ss in series_ids])
    n_cols = len(views) * n_series_max
    if savepath is not None and savepath.endswith('.pdf'):
        raise ValueError(f"Use '.jpg' or '.png' for 'savepath', '.pdf' is tricky to load without 'poppler' installed.")

    # Convert figsize from cm to inches.
    figsize = figsize_to_inches(figsize)

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
            dose_series_id = dose_series_ids[i][series_idx] if isinstance(dose_series_ids[i], list) else dose_series_ids[i]
            dose_series_date = dose_series_dates[i][series_idx] if isinstance(dose_series_dates[i], list) else dose_series_dates[i]
            spacing = spacings[i][series_idx] if isinstance(spacings[i], list) else spacings[i]
            offset = offsets[i][series_idx] if isinstance(offsets[i], list) else offsets[i]
            region_data = region_datas[i][series_idx] if isinstance(region_datas[i], list) else region_datas[i]
            landmark_data = landmark_datas[i][series_idx] if isinstance(landmark_datas[i], list) else landmark_datas[i]
            landmark_data_other = landmark_datas_other[i][series_idx] if isinstance(landmark_datas_other[i], list) else landmark_datas_other[i]
            crop = crops[i][series_idx] if isinstance(crops[i], list) else crops[i]
            centre = centres[i][series_idx] if isinstance(centres[i], list) else centres[i]
            series_id = series_ids[i][series_idx] if isinstance(series_ids[i], list) else series_ids[i]
            study_id = study_ids[i][series_idx] if isinstance(study_ids[i], list) else study_ids[i]
            study_date = study_dates[i][series_idx] if isinstance(study_dates[i], list) else study_dates[i]
            view = views[view_idx] if len(views) > 1 else views[0]

            plot_patient(
                ct_data.shape,
                spacing,
                ax=axs[i][j],
                centre=centre,
                close_figure=False,
                crop=crop,
                ct_data=ct_data,
                dose_data=dose_data,
                dose_series_date=dose_series_date,
                dose_series_id=dose_series_id,
                landmark_data=landmark_data,
                landmark_data_other=landmark_data_other,
                offset=offset,
                pat_id=pat_id,
                region_data=region_data,
                series_id=series_id,
                study_date=study_date,
                study_id=study_id,
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
    offset: Point3D,
    view: Axis) -> Box2D:
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
    min_vox = get_view_xy(min_vox, view)
    max_vox = get_view_xy(max_vox, view)

    return min_vox, max_vox

def __map_regions(
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

def __get_region_crop(
    data: RegionArray,
    crop_margin_mm: float,
    spacing: Spacing3D,
    offset: Point3D,
    view: Axis) -> Box2D:
    # Get region extent.
    ext_min_mm, ext_max_mm = extent(data, spacing, offset)

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
    min_vox = get_view_xy(min_vox, view)
    max_vox = get_view_xy(max_vox, view)

    return min_vox, max_vox

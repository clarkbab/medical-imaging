from typing import *

from mymi.typing import *
from mymi.utils import *

from .patients import plot_single_patient
from .plotting import *

def plot_registration(
    dataset_type: Dataset,
    dataset_fns: Dict[str, Callable],
    dataset: str,
    model: str,
    centre: Optional[Union[LandmarkID, RegionID]] = None,
    crop: Optional[Union[Box2D, LandmarkID, RegionID]] = None,
    crop_margin: float = 100,
    exclude_fixed_pat: Optional[PatientIDs] = None,
    exclude_moving_pat: Optional[PatientIDs] = None,
    fixed_centre: Optional[Union[LandmarkID, RegionID]] = None,
    fixed_crop: Optional[Union[Box2D, LandmarkID, RegionID]] = None,
    fixed_idx: Optional[Union[int, float]] = None,
    fixed_pat: Optional[PatientIDs] = 'all',
    fixed_study: StudyID = 'study_1',
    group: PatientGroups = 'all',
    idx: Optional[Union[int, float]] = None,
    isodoses: Union[float, List[float]] = [],
    labels: Literal['included', 'excluded', 'all'] = 'all',
    landmark: Optional[LandmarkIDs] = 'all',
    loadpath: Optional[str] = None,
    moved_centre: Optional[Union[RegionID]] = None,
    moved_crop: Optional[Union[Box2D, RegionID]] = None,
    moved_idx: Optional[Union[int, float]] = None,
    moving_centre: Optional[Union[LandmarkID, RegionID]] = None,
    moving_crop: Optional[Union[Box2D, LandmarkID, RegionID]] = None,
    moving_idx: Optional[Union[int, float]] = None,
    moving_pat: Optional[PatientIDs] = None,
    moving_study: StudyID = 'study_0',
    region: Optional[RegionIDs] = None,
    region_labels: Optional[Dict[str, str]] = None,
    show_dose: bool = False,
    show_fixed_dose: bool = False,
    show_moved_dose: bool = False,
    show_moving_dose: bool = False,
    **kwargs) -> None:
    if loadpath is not None:
        plot_saved(loadpath)
        return
    set = dataset_type(dataset)
    fixed_pat_ids = set.list_patients(exclude=exclude_fixed_pat, group=group, pat=fixed_pat)
    if moving_pat is None:
        moving_pat_ids = fixed_pat_ids
    else:
        moving_pat_ids = arg_to_list(moving_pat, PatientID, literals={ 'all': set.list_patients(exclude=exclude_moving_pat, group=group) })
        assert len(moving_pat_ids) == len(fixed_pat_ids)

    moving_study_id = moving_study
    fixed_study_id = fixed_study
    isodoses = arg_to_list(isodoses, float)
    show_doses = [show_moving_dose or show_dose, show_fixed_dose or show_dose, show_moved_dose or show_dose]

    moving_ct_datas, fixed_ct_datas, moved_ct_datas = [], [], []
    moving_dose_datas, fixed_dose_datas, moved_dose_datas = [], [], []
    moving_centres, fixed_centres, moved_centres = [], [], []
    moving_crops, fixed_crops, moved_crops = [], [], []
    moving_spacings, fixed_spacings = [], []
    moving_origins, fixed_origins = [], []
    moving_landmarks_datas, fixed_landmarks_datas, moved_landmarks_datas = [], [], []
    moving_regions_datas, fixed_regions_datas, moved_regions_datas = [], [], []
    moving_idxs, fixed_idxs, moved_idxs = [], [], []
    transforms = []

    for i, p in enumerate(fixed_pat_ids):
        moving_pat_id = p if moving_pat_ids is None else moving_pat_ids[i]

        # Load moving and fixed CT and region data.
        # TODO: Expand to multiple study IDs.
        pat_ids = [(moving_pat_id, moving_study_id), (p, fixed_study_id)]
        ct_datas = []
        dose_datas = []
        landmarks_datas = []
        regions_datas = []
        spacings = []
        centres = []
        crops = []
        origins = []
        centres = [moving_centre if moving_centre is not None else centre, fixed_centre if fixed_centre is not None else centre, moved_centre if moved_centre is not None else centre]
        crops = [moving_crop if moving_crop is not None else crop, fixed_crop if fixed_crop is not None else crop, moved_crop if moved_crop is not None else crop]
        idxs = [moving_idx if moving_idx is not None else idx, fixed_idx if fixed_idx is not None else idx, moved_idx if moved_idx is not None else idx]
        for j, (p, s) in enumerate(pat_ids):
            study = set.patient(p).study(s)
            ct_image = study.default_ct
            ct_datas.append(ct_image.data)
            spacings.append(ct_image.spacing)
            origins.append(ct_image.origin)
            if show_doses[j] or len(isodoses) > 0 and study.has_dose:
                # Resample dose data to CT spacing/origin.
                dose_image = study.default_dose
                resample_kwargs = dict(
                    origin=dose_image.origin,
                    output_origin=ct_image.origin,
                    output_size=ct_image.size,
                    output_spacing=ct_image.spacing,
                    spacing=dose_image.spacing,
                )
                dose_data = resample(dose_image.data, **resample_kwargs)
            else:
                dose_data = None
            dose_datas.append(dose_data)
            if landmark is not None:
                landmarks_data = study.landmarks_data(landmark=landmark)
            else:
                landmarks_data = None
            if region is not None:
                regions_data = study.regions_data(labels=labels, region=region)
            else:
                regions_data = None

            c = centres[j] 
            oc = None
            if c is not None:
                if isinstance(c, (LandmarkID, RegionID)):
                    # Load 'centre' data if not already in 'landmarks/regions_data'.
                    if study.has_landmark(c):
                        oc = study.landmarks_data(landmark=centre).iloc[0] if landmarks_data is None or centre not in list(landmarks_data['landmark-id']) else centre
                    elif study.has_region(centre):
                        oc = study.regions_data(region=centre)[centre] if regions_data is None or centre not in regions_data else centre
                    else:
                        raise ValueError(f"Study {study} has no landmark/regions with ID '{centre}' for 'centre'.")
                else:
                    oc = c

            # Load 'crop' data if not already in 'regions_data'.
            crop = crops[j]
            ocrop = None
            if crop is not None:
                if type(crop) == str:
                    if regions_data is None or crop not in regions_data:
                        ocrop = study.regions_data(region=crop)[crop]
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
                if type(oc) == str and oc in region_labels:
                    oc = region_labels[oc] 
                if type(ocrop) == str and ocrop in region_labels:
                    ocrop = region_labels[ocrop]
            
            landmarks_datas.append(landmarks_data)
            regions_datas.append(regions_data)
            centres.append(oc)
            crops.append(ocrop)

        # Load registered data.
        transform, moved_ct_data, moved_regions_data, moved_landmarks_data, moved_dose_data = dataset_fns['load_registration'](dataset, p, model, study=fixed_study, landmark=landmark, load_dose=show_doses[2] or len(isodoses) > 0, moving_pat=moving_pat_id, moving_study=moving_study_id, region=region) 

        # Add moved centre.
        _, _, all_moved_regions_data, all_moved_landmarks_data, _ = dataset_fns['load_registration'](dataset, p, model, study=fixed_study, moving_pat=moving_pat_id, moving_study=moving_study_id, use_patient_coords=False) 
        c = centres[2]
        moved_centre = None
        if c is not None:
            if isinstance(c, (LandmarkID, RegionID)):
                # Load 'centre' data if not already in 'landmarks/regions_data'.
                if all_moved_landmarks_data is not None and c in list(all_moved_landmarks_data['landmark-id']):
                    moved_centre = None
                elif all_moved_regions_data is not None and c in all_moved_regions_data:
                    moved_centre = all_moved_regions_data[c] if moved_regions_data is None or c not in moved_regions_data else c
                else:
                    raise ValueError(f"No moved landmark/regions '{c}' for model={model}, study={fixed_study_id}, moving_study={moving_study_id}. Can't use for 'centre'.")
            else:
                moved_centre = c

        # Load 'moved_crop' data if not already in 'moved_regions_data'.
        crop = crops[2]
        moved_crop = None
        if crop is not None:
            if type(crop) == str:
                if moved_regions_data is None or crop not in moved_regions_data:
                    _, _, crop_regions_data, _, _ = load_reg_fn(dataset, p, model, fixed_study=fixed_study_id, moving_pat=moving_pat_id, moving_study=moving_study, region=crop)
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
        fixed_origins.append(origins[1])
        moving_origins.append(origins[0])
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
        fixed_idxs.append(idxs[1])
        moving_idxs.append(idxs[0])
        moved_idxs.append(moved_idx)
        transforms.append(transform)

    okwargs = dict(
        fixed_centres=fixed_centres,
        fixed_crops=fixed_crops,
        fixed_crop_margin=crop_margin,
        fixed_ct_datas=fixed_ct_datas,
        fixed_dose_datas=fixed_dose_datas,
        fixed_idxs=fixed_idxs,
        fixed_landmarks_datas=fixed_landmarks_datas,
        fixed_origins=fixed_origins,
        fixed_spacings=fixed_spacings,
        fixed_regions_datas=fixed_regions_datas,
        isodoses=isodoses,
        moved_centres=moved_centres,
        moved_crops=moved_crops,
        moved_crop_margin=crop_margin,
        moved_ct_datas=moved_ct_datas,
        moved_dose_datas=moved_dose_datas,
        moved_idxs=moved_idxs,
        moved_landmarks_datas=moved_landmarks_datas,
        moved_regions_datas=moved_regions_datas,
        moving_centres=moving_centres,
        moving_crops=moving_crops,
        moving_crop_margin=crop_margin,
        moving_ct_datas=moving_ct_datas,
        moving_dose_datas=moving_dose_datas,
        moving_idxs=moving_idxs,
        moving_landmarks_datas=moving_landmarks_datas,
        moving_origins=moving_origins,
        moving_spacings=moving_spacings,
        moving_regions_datas=moving_regions_datas,
        show_fixed_dose=show_doses[1],
        show_moved_dose=show_doses[2],
        show_moving_dose=show_doses[0],
        transforms=transforms,
    )
    plot_registrations_matrix(fixed_pat_ids, fixed_study_id, moving_pat_ids, moving_study_id, **okwargs, **kwargs)

def plot_registrations_matrix(
    fixed_pat_ids: Sequence[PatientID],
    fixed_studies: Sequence[StudyID],
    moving_pat_ids: Sequence[PatientID],
    moving_studies: Sequence[StudyID],
    fixed_centres: Sequence[Optional[Union[str, np.ndarray]]] = [],             # Uses 'fixed_regions_data' if 'str', else uses 'np.ndarray'.
    fixed_crops: Sequence[Optional[Union[str, np.ndarray, Box2D]]] = [],    # Uses 'fixed_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    fixed_ct_datas: Sequence[Optional[CtImageArray]] = [],
    fixed_dose_datas: Sequence[Optional[DoseImageArrays]] = [],
    fixed_idxs: Sequence[Optional[Union[int, float]]] = [],
    fixed_landmarks_datas: Sequence[Optional[LandmarksFrame]] = [],
    fixed_origins: Sequence[Optional[Point3D]] = [],
    fixed_regions_datas: Sequence[Optional[np.ndarray]] = [],
    fixed_spacings: Sequence[Optional[Spacing3D]] = [],
    figsize: Tuple[float, float] = (16, 4),     # Width always the same, height is based on a single row.
    moved_centres: Sequence[Optional[Union[str, LabelArray]]] = [],             # Uses 'moved_regions_data' if 'str', else uses 'np.ndarray'.
    moved_crops: Sequence[Optional[Union[str, LabelArray, Box2D]]] = [],    # Uses 'moved_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moved_ct_datas: Sequence[Optional[CtImageArray]] = [],
    moved_dose_datas: Sequence[Optional[DoseImageArrays]] = [],
    moved_idxs: Sequence[Optional[Union[int, float]]] = [],
    moved_landmarks_datas: Sequence[Optional[LandmarksFrame]] = [],
    moved_regions_datas: Sequence[Optional[RegionArrays]] = [],
    moving_centres: Sequence[Optional[Union[str, LabelArray]]] = [],             # Uses 'moving_regions_data' if 'str', else uses 'np.ndarray'.
    moving_crops: Sequence[Optional[Union[str, LabelArray, Box2D]]] = [],    # Uses 'moving_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moving_ct_datas: Sequence[Optional[CtImageArray]] = [],
    moving_dose_datas: Sequence[Optional[DoseImageArrays]] = [],
    moving_idxs: Sequence[Optional[Union[int, float]]] = [],
    moving_landmarks_datas: Sequence[Optional[LandmarksFrame]] = [],
    moving_origins: Sequence[Optional[Point3D]] = [],
    moving_regions_datas: Sequence[Optional[RegionArrays]] = [],
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
            fixed_origin=fixed_origins[i],
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
            moving_origin=moving_origins[i],
            moving_regions_data=moving_regions_datas[i],
            moving_spacing=moving_spacings[i],
            show_fixed=show_fixed,
            show_grid=show_grid,
            show_moving=show_moving,
            transform=transforms[i],
        )
        plot_single_registration(fixed_pat_ids[i], fixed_studies[i], moving_pat_ids[i], moving_studies[i], **okwargs, **kwargs)

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

def plot_single_registration(
    fixed_pat: PatientID,
    fixed_study: StudyID,
    moving_pat: PatientID,
    moving_study: StudyID,
    alpha_region: float = 0.5,
    aspect: Optional[float] = None,
    axs: Optional[mpl.axes.Axes] = None,
    extent_of: Optional[Union[Tuple[Union[str, np.ndarray], Extrema], Tuple[Union[str, np.ndarray], Extrema, Axis]]] = None,          # Tuple of object to crop to (uses 'regions_data' if 'str', else 'np.ndarray') and min/max of extent.
    fixed_centre: Optional[Union[LandmarkSeries, LandmarkID, RegionArray, RegionID]] = None,
    fixed_crop: Optional[Union[str, np.ndarray, Box2D]] = None,    # Uses 'fixed_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    fixed_crop_margin: float = 100,                                       # Applied if cropping to 'fixed_regions_data' or 'np.ndarray'
    fixed_ct_data: Optional[CtImageArray] = None,
    fixed_dose_data: Optional[DoseImageArray] = None,
    fixed_idx: Optional[int] = None,
    fixed_landmarks_data: Optional[LandmarksFrame] = None,
    fixed_origin: Optional[Point3D] = None,
    fixed_regions_data: Optional[np.ndarray] = None,
    fixed_spacing: Optional[Spacing3D] = None,
    figsize: Tuple[float, float] = (30, 10),
    fontsize: int = 12,
    latex: bool = False,
    match_landmarks: bool = True,
    moved_centre: Optional[Union[RegionArray, RegionID]] = None,
    moved_crop: Optional[Union[str, np.ndarray, Box2D]] = None,    # Uses 'moved_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moved_crop_margin: float = 100,                                       # Applied if cropping to 'moved_regions_data' or 'np.ndarray'
    moved_ct_data: Optional[CtImageArray] = None,
    moved_dose_data: Optional[DoseImageArray] = None,
    moved_idx: Optional[int] = None,
    moved_landmarks_colour: str = 'red',
    moved_landmarks_data: Optional[LandmarkIDs] = None,
    moved_regions_data: Optional[np.ndarray] = None,
    moved_use_fixed_idx: bool = True,
    moving_centre: Optional[Union[RegionArray, RegionID]] = None,
    moving_crop: Optional[Union[str, np.ndarray, Box2D]] = None,    # Uses 'moving_regions_data' if 'str', else uses 'np.ndarray' or crop co-ordinates.
    moving_crop_margin: float = 100,                                       # Applied if cropping to 'moving_regions_data' or 'np.ndarray'
    moving_ct_data: Optional[CtImageArray] = None,
    moving_dose_data: Optional[DoseImageArray] = None,
    moving_idx: Optional[int] = None,
    moving_landmarks_data: Optional[LandmarkIDs] = None,
    moving_origin: Optional[Point3D] = None,
    moving_regions_data: Optional[np.ndarray] = None,
    moving_spacing: Optional[Spacing3D] = None,
    n_landmarks: Optional[int] = None,
    show_fixed: bool = True,
    show_fixed_dose: bool = False,
    show_grid: bool = True,
    show_legend: bool = False,
    show_moved_dose: bool = False,
    show_moved_landmarks: bool = True,
    show_moving: bool = True,
    show_moving_dose: bool = False,
    show_moving_landmarks: bool = True,
    show_region_overlay: bool = True,
    transform: Optional[Union[itk.Transform, sitk.Transform]] = None,
    transform_format: Literal['itk', 'sitk'] = 'sitk',
    view: Axis = 0,
    **kwargs) -> None:
    if n_landmarks is not None and match_landmarks:
        # Ensure that the n "moving/moved" landmarks targeted by "n_landmarks" are
        # are the same as the n "fixed" landmarks. 

        # Get n fixed landmarks.
        fixed_landmarks_data['dist'] = np.abs(fixed_landmarks_data[view] - fixed_idx)
        fixed_landmarks_data = fixed_landmarks_data.sort_values('dist')
        fixed_landmarks_data = fixed_landmarks_data.iloc[:n_landmarks]
        fixed_landmarks = fixed_landmarks_data['landmark-id'].tolist()

        # Get moving/moved landmarks.
        moving_landmarks_data = moving_landmarks_data[moving_landmarks_data['landmark-id'].isin(fixed_landmarks)]
        moved_landmarks_data = moved_landmarks_data[moved_landmarks_data['landmark-id'].isin(fixed_landmarks)]

        okwargs = dict(
            colour=moved_landmarks_colour,
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
    origins = [moving_origin if show_moving else hidden, fixed_origin if show_fixed else hidden, fixed_origin]
    origins = [c for c in origins if not (isinstance(c, str) and c == hidden)]
    if moved_ct_data is None:
        raise ValueError(f"No moved CT data for registration for patient {moving_pat}->{fixed_pat}, study {moving_study}->{fixed_study}.")
    sizes = [moving_ct_data.shape if show_moving else hidden, fixed_ct_data.shape if show_fixed else hidden, moved_ct_data.shape]
    sizes = [c for c in sizes if not (isinstance(c, str) and c == hidden)]
    centres = [moving_centre if show_moving else hidden, fixed_centre if show_fixed else hidden, moved_centre]
    centres = [c for c in centres if not (isinstance(c, str) and c == hidden)]
    crops = [moving_crop if show_moving else hidden, fixed_crop if show_fixed else hidden, moved_crop]
    crops = [c for c in crops if not (isinstance(c, str) and c == hidden)]
    crop_margins = [moving_crop_margin if show_moving else hidden, fixed_crop_margin if show_fixed else hidden, moved_crop_margin]
    crop_margins = [c for c in crop_margins if not (isinstance(c, str) and c == hidden)]
    pat_ids = [moving_pat if show_moving else hidden, fixed_pat if show_fixed else hidden, f'{moving_pat} (moved)']
    pat_ids = [c for c in pat_ids if not (isinstance(c, str) and c == hidden)]
    studies = [moving_study if show_moving else hidden, fixed_study if show_fixed else hidden, f'{moving_study} (moved)']
    studies = [c for c in studies if not (isinstance(c, str) and c == hidden)]
    landmarks_datas = [moving_landmarks_data if show_moving else hidden, fixed_landmarks_data if show_fixed else hidden, None]
    if not show_moving_landmarks:
        # Hide moving landmarks, but we still need 'moving_landmarks_data' != None for the
        # "moved" landmark plotting code.
        landmarks_datas[0] = None
    landmarks_datas = [l for l in landmarks_datas if not (isinstance(l, str) and l == hidden)]
    regions_datas = [moving_regions_data if show_moving else hidden, fixed_regions_data if show_fixed else hidden, moved_regions_data]
    regions_datas = [c for c in regions_datas if not (isinstance(c, str) and c == hidden)]
    if moved_use_fixed_idx:
        # Plot patients resolves the fixed idx based on parameters (e.g. idx, centre).
        # We need to resolve the fixed idx here for the moved image.
        resolved_fixed_idx = get_idx(sizes[1], view, centre=centres[1], idx=fixed_idx, landmarks_data=fixed_landmarks_data, origin=origins[1], regions_data=fixed_regions_data, spacing=spacings[1])
    idxs = [moving_idx if show_moving else hidden, fixed_idx if show_fixed else hidden, resolved_fixed_idx if moved_use_fixed_idx else moved_idx]
    idxs = [c for c in idxs if not (isinstance(c, str) and c == hidden)]
    show_doses = [show_moving_dose, show_fixed_dose, show_moved_dose]

    n_rows = 2 if show_grid else 1
    n_cols = show_moving + show_fixed + 1
    axs = arg_to_list(axs, mpl.axes.Axes)
    if axs is None:
        figsize_width, figsize_height = figsize
        figsize_height = n_rows * figsize_height
        figsize = (figsize_width, figsize_height)
        # How do I remove vertical spacing???
        fig = plt.figure(figsize=figsize_to_inches(figsize))
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
        # Add moved landmarks.
        lm_other = moved_landmarks_data if show_moving and i == 0 else None
        okwargs = dict(
            ax=axs[i],
            centre=centres[i],
            crop=crops[i],
            ct_data=ct_datas[i],
            dose_data=dose_datas[i],
            idx=idxs[i],
            landmarks_data=landmarks_datas[i],
            landmarks_data_other=lm_other,
            n_landmarks=n_landmarks,
            origin=origins[i],
            pat=pat_ids[i],
            regions_data=regions_datas[i],
            show_dose=show_doses[i],
            show_legend=show_legend,
            show_title_study=False,
            study=studies[i],
            view=view,
        )
        plot_single_patient(ct_datas[i].shape, spacings[i], **okwargs, **kwargs)

    # # Add moved landmarks to moving image.
    # if show_moving and show_moved_landmarks and moved_landmarks_data is not None:
    #     moving_idx = get_idx(sizes[0], view, centre=centres[0], idx=idxs[0], landmarks_data=moving_landmarks_data, spacing=spacings[0], origin=origins[0])
    #     okwargs = dict(
    #         colour=moved_landmarks_colour,
    #         dose_data=dose_datas[0],
    #         zorder=0,
    #     )
    #     plot_landmarks_data(moved_landmarks_data, axs[0], moving_idx, sizes[0], spacings[0], origins[0], view, **okwargs, **kwargs)

    # Add fixed landmarks to moved image.
    if fixed_landmarks_data is not None:
        # Resolve the moved idx - usually handled by 'plot_patients'.
        moved_idx = resolved_fixed_idx if moved_use_fixed_idx else get_idx(sizes[2], view, centre=centres[2], idx=idxs[2], landmarks_data=moving_landmarks_data, spacing=spacings[2], origin=origins[2])
        okwargs = dict(
            dose_data=dose_datas[2],
            zorder=0,
        )
        plot_landmarks_data(fixed_landmarks_data, axs[2], moved_idx, sizes[2], spacings[2], origins[2], view, **okwargs, **kwargs)

    if show_grid:
        # Plot moving grid.
        include = [True] * 3
        include[view] = False
        moving_grid = __create_grid(moving_ct_data.shape, moving_spacing, include=include)
        moving_idx = get_idx(sizes[0], view, centre=centres[0], idx=idxs[0], landmarks_data=moving_landmarks_data, spacing=spacings[0], origin=origins[0])
        if show_moving:
            grid_slice, _ = get_view_slice(moving_grid, moving_idx, view)
            aspect = get_aspect(view, moving_spacing)
            origin = get_view_origin(view)[1]
            axs[n_cols].imshow(grid_slice, aspect=aspect, cmap='gray', origin=origin)

        # Plot moved grid.
        if transform_format == 'itk':
            # When ITK loads nifti images, it reversed direction/origin for x/y axes.
            # This is an issue as our code doesn't use directions, it assumes a positive direction matrix.
            # I don't know how to reverse x/y axes with ITK transforms, so we have to do it with 
            # images before applying the transform.
            moved_grid = itk_transform_image(moving_grid, transform, fixed_ct_data.shape, origin=moving_origin, output_origin=fixed_origin, output_spacing=fixed_spacing, spacing=moving_spacing)
        elif transform_format == 'sitk':
            moved_grid = resample(moving_grid, origin=moving_origin, output_origin=fixed_origin, output_size=fixed_ct_data.shape, output_spacing=fixed_spacing, spacing=moving_spacing, transform=transform)
        grid_slice, _ = get_view_slice(moved_grid, moving_idx, view)
        aspect = get_aspect(view, fixed_spacing)
        origin = get_view_origin(view)[1]
        axs[2 * n_cols - 1].imshow(grid_slice, aspect=aspect, cmap='gray', origin=origin)

        if show_region_overlay:
            # Plot fixed/moved regions.
            aspect = get_aspect(view, fixed_spacing)
            okwargs = dict(
                alpha=alpha_region,
                crop=fixed_crop,
                view=view,
            )
            background, _ = get_view_slice(np.zeros(shape=fixed_ct_data.shape), idxs[2], view)
            if fixed_crop is not None:
                background = crop_fn(background, transpose_box(fixed_crop), use_patient_coords=False)
            axs[2 * n_cols - 2].imshow(background, cmap='gray', aspect=aspect, interpolation='none', origin=get_view_origin(view)[1])
            if fixed_regions_data is not None:
                plot_regions_data(fixed_regions_data, axs[2 * n_cols - 2], idxs[2], aspect, **okwargs)
            if moved_regions_data is not None:
                plot_regions_data(moved_regions_data, axs[2 * n_cols - 2], idxs[2], aspect, **okwargs)

    if show_figure:
        plt.show()
        plt.close()

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
    origin = [-1, -1, -1]
    grid_spacing_voxels = np.array(grid_spacing) / spacing

    for axis in range(3):
        if include[axis]:
            # Get line positions.
            line_idxs = [i for i in list(np.arange(grid.shape[axis])) if int(np.floor((i - origin[axis]) % grid_spacing_voxels[axis])) in tuple(range(linewidth))]

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

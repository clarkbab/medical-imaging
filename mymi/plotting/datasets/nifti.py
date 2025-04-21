from collections.abc import Iterable
import numpy as np
import re
from typing import *
from wand import image

from mymi.datasets import NiftiDataset
from mymi.gradcam.dataset.nifti import load_multi_segmenter_heatmap
from mymi.predictions.datasets.nifti import load_registration, load_segmenter_predictions
from mymi.typing import *
from mymi.utils import *

from ..plotting import apply_region_labels, plot_heatmap as plot_heatmap_base, plot_histograms, plot_loaded, plot_patients_matrix, plot_registrations as plot_registrations_base, plot_segmenter_predictions as plot_segmenter_predictions_base

MODEL_SELECT_PATTERN = r'^model:([0-9]+)$'
MODEL_SELECT_PATTERN_MULTI = r'^model(:([0-9]+))?:([a-zA-Z_]+)$'

def plot_dataset_histogram(
    dataset: str,
    n_pats: Optional[int] = None,
    pat_ids: Optional[PatientIDs] = None,
    **kwargs) -> None:
    set = NiftiDataset(dataset)
    if n_pats is not None:
        assert pat_ids is None
        pat_ids = set.list_patients()
        pat_ids = pat_ids[:n_pats]
    ct_data = [set.patient(pat_id).ct_data for pat_id in pat_ids]
    ct_data = [c.flatten() for c in ct_data]
    ct_data = np.concatenate(ct_data)
    plot_histograms(ct_data, **kwargs)

def plot_patient_histograms(
    dataset: str,
    pat_ids: PatientIDs = 'all',
    study_ids: StudyIDs = 'all',
    **kwargs) -> None:
    set = NiftiDataset(dataset)
    pat_ids = arg_to_list(pat_ids, PatientID, literals={ 'all': set.list_patients })
    n_rows = len(pat_ids)

    # Get n_cols.
    n_cols = 0
    study_idses = []
    for p in pat_ids:
        pat = set.patient(p)
        study_ids = arg_to_list(study_ids, StudyID, literals={ 'all': pat.list_studies })
        study_idses.append(study_ids)
        if len(study_ids) > n_cols:
            n_cols = len(study_ids)
    
    _, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    for row_axs, p, ss in zip(axs, pat_ids, study_idses):
        pat = set.patient(p)
        for col_ax, s in zip(row_axs, ss):
            study = pat.study(s)
            ct_data = study.ct_data.flatten()
            title = f"{p}:{s}"
            plot_histograms(ct_data, axs=col_ax, title=title, **kwargs)

def plot_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    target_region: str,
    layer: Union[int, str],
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    model_region: Optional[Regions] = None,
    pred_region: Optional[Regions] = None,
    region: Optional[Regions] = None,
    show_ct: bool = True,
    **kwargs) -> None:
    layer = str(layer)

    # Load data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data if show_ct else None
    region_data = pat.region_data(region=region) if region is not None else None
    spacing = pat.ct_spacing

    # Load heatmap.
    heatmap = load_multi_segmenter_heatmap(dataset, pat_id, model, target_region, layer)
    heatmap = np.maximum(heatmap, 0)

    # Load predictions.
    if pred_region is not None:
        # Load segmenter prediction.
        if model_region is None:
            raise ValueError(f"'model_region' is required to load prediction for 'pred_region={pred_region}'.")
        pred_data = load_multi_segmenter_prediction_dict(dataset, pat_id, model, model_region, region=pred_region)
        pred_data = dict((f'pred:{r}', p_data) for r, p_data in pred_data.items())
    else:
        pred_data = None

    if centre is not None:
        if isinstance(centre, str):
            match = re.search(MODEL_SELECT_PATTERN_MULTI, centre)
            if match is not None:
                assert match.group(2) is None
                region_centre = match.group(3)
                centre_tmp = centre
                if pred_data is None:
                    if model_region is None:
                        raise ValueError(f"'model_region' is required to load prediction for 'centre={centre}'.")
                    pred_data_centre = load_multi_segmenter_prediction_dict(dataset, pat_id, model, model_region) 
                else:
                    pred_data_centre = pred_data
                centre = pred_data_centre[region_centre]
                if centre.sum() == 0:
                    raise ValueError(f"Got empty prediction for 'centre={centre_tmp}, please provide 'idx' instead.")
            elif region_data is None or centre not in region_data:
                centre = pat.region_data(region=centre)[centre]

    if crop is not None:
        if isinstance(crop, str):
            if region_data is None or crop not in region_data:
                crop = pat.region_data(region=crop)[crop]
    
    # Plot.
    plot_id = f"{dataset}:{pat_id}"
    plot_heatmap_base(plot_id, heatmap, spacing, centre=centre, crop=crop, ct_data=ct_data, pred_data=pred_data, region_data=region_data, **kwargs)

def plot_patients(
    dataset: str,
    pat_ids: Union[PatientIDs, int] = 'all',
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    landmarks: Optional[Landmarks] = None,
    loadpath: Optional[str] = None,
    region_labels: Dict[str, str] = {},
    regions: Optional[Regions] = None,
    study_ids: Optional[Union[StudyID, List[StudyID], Literal['all']]] = None,
    **kwargs) -> Optional[image.Image]:
    if loadpath is not None:
        return plot_loaded(loadpath)

    set = NiftiDataset(dataset)
    if isinstance(pat_ids, int):
        pat_ids = set.list_patients()[:pat_ids]
    else:
        pat_ids = arg_to_list(pat_ids, PatientID, literals={ 'all': set.list_patients })

    # Load CT data.
    study_idses = []
    plot_ids = []
    ct_datas = []
    spacings = []
    region_datas = []
    landmark_datas = []
    dose_datas = []
    centre_datas = []
    crop_datas = []
    for p in pat_ids:
        # Get study IDs.
        pat = set.patient(p)
        study_ids = arg_to_list(study_ids, StudyID, literals={ 'all': pat.list_studies })
        if study_ids is None:
            study_ids = [pat.default_study.id]
        study_idses.append(study_ids)

        # Add data for each study.
        ct_data = []
        spacing = []
        plot_id = []
        region_data = []
        landmark_data = []
        centre_data = []
        crop_data = []
        for s in study_ids:
            study = pat.study(s)
            plot_id.append(f'{p}:{truncate_str(s, max_chars=10)}')
            ct_data.append(study.ct_data)
            spacing.append(study.ct_spacing)
            if regions is not None:
                rdata = study.region_data(regions=regions, **kwargs)
            else:
                rdata = None
            if landmarks is not None:
                ldata = study.landmark_data(landmarks=landmarks, use_image_coords=True, **kwargs)
            else:
                ldata = None
            region_data.append(rdata)
            landmark_data.append(ldata)

            # If 'centre' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
            c = None
            if centre is not None:
                if isinstance(centre, str):
                    if study.has_regions(centre) and rdata is not None and centre not in rdata:
                        c = study.region_data(regions=centre)[centre]
                    elif study.has_landmark(centre) and ldata is not None and centre not in ldata['landmark-id']:
                        c = study.landmark_data(landmarks=centre)
            centre_data.append(c)

            # If 'crop' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
            c = None
            if crop is not None:
                if isinstance(crop, str):
                    if study.has_regions(crop) and rdata is not None and crop not in rdata:
                        c = study.region_data(regions=crop)[crop]
                    elif study.has_landmark(crop) and ldata is not None and crop not in ldata['landmark-id']:
                        c = study.landmark_data(landmarks=crop)
            crop_data.append(c)

        # Apply region labels.
        # This should maybe be moved to base 'plot_patient'? All of the dataset-specific plotting functions
        # use this. Of course 'plot_patient' API would change to include 'region_labels' as an argument.
        for i in range(len(study_ids)):
            region_data[i], centre_data[i], crop_data[i] = apply_region_labels(region_labels, region_data[i], centre_data[i], crop_data[i])

        # Condense if only one study ID is requested.    
        if len(study_ids) == 1:
            ct_data = ct_data[0]
            spacing = spacing[0]
            plot_id = plot_id[0]
            region_data = region_data[0]
            landmark_data = landmark_data[0]
            centre_data = centre_data[0]
            crop_data = crop_data[0]

        ct_datas.append(ct_data)
        spacings.append(spacing)
        plot_ids.append(plot_id)
        region_datas.append(region_data)
        landmark_datas.append(landmark_data)
        centre_datas.append(centre_data)
        crop_datas.append(crop_data)

    # Plot.
    okwargs = dict(
        centres=centre_datas,
        crops=crop_datas,
        ct_datas=ct_datas,
        dose_datas=dose_datas,
        landmark_datas=landmark_datas,
        region_datas=region_datas,
        study_ids=study_idses,
    )
    plot_patients_matrix(plot_ids, spacings, **okwargs, **kwargs)

def plot_moved(
    dataset: str,
    pat_id: PatientID,
    moving_id: StudyID,
    fixed_id: StudyID,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    region_labels: Dict[str, str] = {},
    regions: Optional[Regions] = None,
    regions_ignore_missing: bool = True,
    show_dose: bool = False,
    study_id: Optional[StudyID] = None,
    **kwargs) -> None:

    # Load moved CT.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    moved_ct = load_
    ct_data = study.ct_data
    spacing = study.ct_spacing

    # Load region data.
    if regions is not None:
        region_data = study.region_data(regions=regions, **kwargs)
    else:
        region_data = None

    # Load dose data.
    dose_data = study.dose_data if show_dose else None

    # Load 'centre' data if not in 'region_data'. This will ultimately be a np.ndarray.
    if centre is not None:
        if type(centre) == str:
            if region_data is None or centre not in region_data:
                centre = study.region_data(region=centre)[centre]

    # Load 'crop' data if not in 'region_data'. This will ultimately be a np.ndarray.
    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = study.region_data(region=crop)[crop]

    # Apply region labels.
    # This should maybe be moved to base 'plot_patient'? All of the dataset-specific plotting functions
    # use this. Of course 'plot_patient' API would change to include 'region_labels' as an argument.
    region_data, centre, crop = apply_region_labels(region_labels, region_data, centre, crop)

    # Plot.
    plot_id = f"{dataset}:{pat_id}:{study.id}"
    plot_patients_matrix(plot_id, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, dose_data=dose_data, region_data=region_data, **kwargs)

def plot_registrations(
    dataset: str,
    model: str,
    centre: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[str, List[str]]] = None,
    crop_margin: float = 100,
    fixed_pat_ids: Optional[PatientIDs] = None,
    fixed_study_id: StudyID = 'study_1',
    idx: Optional[Union[int, float, List[Union[int, float]]]] = None,
    labels: Literal['included', 'excluded', 'all'] = 'all',
    landmarks: Optional[Landmarks] = 'all',
    loadpath: Optional[str] = None,
    moving_pat_ids: Optional[PatientIDs] = None,
    moving_study_id: StudyID = 'study_0',
    regions: Optional[Regions] = 'all',
    region_labels: Optional[Dict[str, str]] = None,
    splits: Optional[Splits] = None,
    **kwargs) -> None:
    if loadpath is not None:
        return plot_loaded(loadpath)
    set = NiftiDataset(dataset)
    if splits is not None:
        fixed_pat_ids = set.list_patients(splits=splits)
        moving_pat_ids = fixed_pat_ids
    else:
        if fixed_pat_ids is None:
            raise ValueError("Must provide 'fixed_pat_ids' or 'splits' to plot registrations.")
        fixed_pat_ids = arg_to_list(fixed_pat_ids, PatientID, literals={ 'all': set.list_patients })
        if moving_pat_ids is None:
            moving_pat_ids = fixed_pat_ids
        else:
            moving_pat_ids = arg_to_list(moving_pat_ids, PatientID, literals={ 'all': set.list_patients })

    fixed_study_ids = arg_broadcast(fixed_study_id, len(fixed_pat_ids))
    moving_study_ids = arg_broadcast(moving_study_id, len(moving_pat_ids))

    moving_ct_datas, fixed_ct_datas, moved_ct_datas = [], [], []
    moving_centres, fixed_centres, moved_centres = [], [], []
    moving_crops, fixed_crops, moved_crops = [], [], []
    moving_spacings, fixed_spacings, moved_spacings = [], [], []
    moving_offsets, fixed_offsets, moved_offsets = [], [], []
    moving_landmark_datas, fixed_landmark_datas, moved_landmark_datas = [], [], []
    moving_region_datas, fixed_region_datas, moved_region_datas = [], [], []
    moving_idxs, fixed_idxs, moved_idxs = [], [], []
    transforms = []

    for i, p in enumerate(fixed_pat_ids):
        moving_pat_id = p if moving_pat_ids is None else moving_pat_ids[i]

        # Load moving and fixed CT and region data.
        ids = [(moving_pat_id, moving_study_id), (p, fixed_study_id)]
        ct_datas = []
        landmark_datas = []
        region_datas = []
        sizes = []
        spacings = []
        centres = []
        crops = []
        offsets = []
        centres_broad = arg_broadcast(centre, 3)
        crops_broad = arg_broadcast(crop, 3)
        idxs_broad = arg_broadcast(idx, 3)
        for j, (p, s) in enumerate(ids):
            study = set.patient(p).study(s)
            ct_data = study.ct_data
            ct_datas.append(ct_data)
            if landmarks is not None:
                landmark_data = study.landmark_data(landmarks=landmarks, use_image_coords=True)
            else:
                landmark_data = None
            if regions is not None:
                region_data = study.region_data(labels=labels, regions=regions, regions_ignore_missing=True)
            else:
                region_data = None
            sizes.append(study.ct_size)
            spacings.append(study.ct_spacing)
            offsets.append(study.ct_offset)

            # Load 'centre' data if not already in 'region_data'.
            centre = centres_broad[j]
            ocentre = None
            if centre is not None:
                if type(centre) == str:
                    if region_data is None or centre not in region_data:
                        ocentre = study.region_data(regions=centre)[centre]
                    else:
                        ocentre = centre
                else:
                    ocentre = centre

            # Load 'crop' data if not already in 'region_data'.
            crop = crops_broad[j]
            ocrop = None
            if crop is not None:
                if type(crop) == str:
                    if region_data is None or crop not in region_data:
                        ocrop = study.region_data(regions=crop)[crop]
                    else:
                        ocrop = crop
                else:
                    ocrop = crop

            # Map region names.
            if region_labels is not None:
                # Rename regions.
                for o, n in region_labels.items():
                    moving_region_data[n] = moving_region_data.pop(o)

                # Rename 'centre' and 'crop' keys.
                if type(ocentre) == str and ocentre in region_labels:
                    ocentre = region_labels[ocentre] 
                if type(ocrop) == str and ocrop in region_labels:
                    ocrop = region_labels[ocrop]
            
            landmark_datas.append(landmark_data)
            region_datas.append(region_data)
            centres.append(ocentre)
            crops.append(ocrop)

        # Load registered data.
        moved_ct_data, transform, moved_region_data, moved_landmark_data = load_registration(dataset, p, model, fixed_study_id=fixed_study_id, landmarks=landmarks, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, regions=regions, regions_ignore_missing=True, use_image_coords=True) 

        # Load 'moved_centre' data if not already in 'moved_region_data'.
        centre = centres_broad[2]
        moved_centre = None
        if centre is not None:
            if type(centre) == str:
                if moved_region_data is None or centre not in moved_region_data:
                    _, _, centre_region_data, _ = load_registration(dataset, p, model, fixed_study_id=fixed_study_id, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, regions=centre)
                    moved_centre = centre_region_data[centre]
                else:
                    moved_centre = centre
            else:
                moved_centre = centre

        # Load 'moved_crop' data if not already in 'moved_region_data'.
        crop = crops_broad[2]
        moved_crop = None
        if crop is not None:
            if type(crop) == str:
                if moved_region_data is None or crop not in moved_region_data:
                    _, _, crop_region_data, _ = load_registration(dataset, p, model, fixed_study_id=fixed_study_id, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, regions=crop)
                    moved_crop = crop_region_data[crop]
                else:
                    moved_crop = crop
            else:
                moved_crop = crop

        # Rename moved labels.
        if region_labels is not None:
            # Rename regions.
            for o, n in region_labels.items():
                moved_region_data[n] = moved_region_data.pop(o)

            # Rename 'centre' and 'crop' keys.
            if type(moved_centre) == str and moved_centre in region_labels:
                moved_centre = region_labels[moved_centre] 
            if type(moved_crop) == str and moved_crop in region_labels:
                moved_crop = region_labels[moved_crop]

        # Add to main lists.
        fixed_ct_datas.append(ct_datas[1])
        moving_ct_datas.append(ct_datas[0])
        moved_ct_datas.append(moved_ct_data)
        fixed_centres.append(centres[1])
        moving_centres.append(centres[0])
        moved_centres.append(moved_centre)
        fixed_crops.append(crops[1])
        moving_crops.append(crops[0])
        moved_crops.append(moved_crop)
        fixed_spacings.append(spacings[1])
        moving_spacings.append(spacings[0])
        moved_spacings.append(spacings[1])
        fixed_offsets.append(offsets[1])
        moving_offsets.append(offsets[0])
        moved_offsets.append(offsets[1])
        fixed_landmark_datas.append(landmark_datas[1])
        moving_landmark_datas.append(landmark_datas[0])
        moved_landmark_datas.append(moved_landmark_data)
        fixed_region_datas.append(region_datas[1])
        moving_region_datas.append(region_datas[0])
        moved_region_datas.append(moved_region_data)
        fixed_idxs.append(idxs_broad[1])
        moving_idxs.append(idxs_broad[0])
        moved_idxs.append(idxs_broad[2])
        transforms.append(transform)

    okwargs = dict(
        fixed_centres=fixed_centres,
        fixed_crops=fixed_crops,
        fixed_crop_margin=crop_margin,
        fixed_ct_datas=fixed_ct_datas,
        fixed_idxs=fixed_idxs,
        fixed_landmark_datas=fixed_landmark_datas,
        fixed_offsets=fixed_offsets,
        fixed_spacings=fixed_spacings,
        fixed_region_datas=fixed_region_datas,
        moved_centres=moved_centres,
        moved_crops=moved_crops,
        moved_crop_margin=crop_margin,
        moved_ct_datas=moved_ct_datas,
        moved_idxs=moved_idxs,
        moved_landmark_datas=moved_landmark_datas,
        moved_offsets=fixed_offsets,
        moved_region_datas=moved_region_datas,
        moving_centres=moving_centres,
        moving_crops=moving_crops,
        moving_crop_margin=crop_margin,
        moving_ct_datas=moving_ct_datas,
        moving_idxs=moving_idxs,
        moving_landmark_datas=moving_landmark_datas,
        moving_offsets=moving_offsets,
        moving_spacings=moving_spacings,
        moving_region_datas=moving_region_datas,
        transforms=transforms,
    )
    plot_registrations_base(fixed_pat_ids, fixed_study_ids, moving_pat_ids, moving_study_ids, **okwargs, **kwargs)

def plot_segmenter_predictions(
    dataset: str,
    pat_id: str,
    model: str,
    centre: Optional[str] = None,
    crop: Optional[str] = None,
    regions: Regions = 'all',
    regions_model: Regions = 'all',
    study_id: str = 'study_0',
    **kwargs) -> None:
    
    # Load data.
    set = NiftiDataset(dataset)
    study = set.patient(pat_id).study(study_id)
    ct_data = study.ct_data
    spacing = study.ct_spacing
    region_data = study.region_data(regions=regions)

    # Load predictions.
    pred_data = load_segmenter_predictions(dataset, pat_id, model, regions=regions_model, study_id=study_id)

    # Only handle centre of ground truth - not pred.
    if isinstance(centre, str):
        assert isinstance(centre, str)
        if region_data is None or centre not in region_data:
            centre = study.region_data(regions=centre)[centre]

    if isinstance(crop, str):
        assert isinstance(crop, str)
        if region_data is None or crop not in region_data:
            crop = study.region_data(regions=crop)[crop]
    
    # Plot.
    okwargs = dict(
        centre=centre,
        crop=crop,
        ct_data=ct_data,
        region_data=region_data,
        **kwargs
    )
    plot_segmenter_predictions_base(pat_id, spacing, pred_data, **okwargs)

from collections.abc import Iterable
import numpy as np
import re
from typing import *

from mymi.datasets import NiftiDataset
from mymi.gradcam.dataset.nifti import load_multi_segmenter_heatmap
from mymi.predictions.datasets.nifti import load_registration
from mymi.typing import *
from mymi.utils import *

from ..plotting import apply_region_labels
from ..plotting import plot_heatmap as plot_heatmap_base
from ..plotting import plot_histogram as plot_histogram_base
from ..plotting import plot_segmenter_predictions as plot_segmenter_predictions_base
from ..plotting import plot_patients_matrix
from ..plotting import plot_registration as plot_registration_base

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
    plot_histogram_base(ct_data, **kwargs)

def plot_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    target_region: str,
    layer: Union[int, str],
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    model_region: Optional[PatientRegions] = None,
    pred_region: Optional[PatientRegions] = None,
    region: Optional[PatientRegions] = None,
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
    pat_ids: PatientIDs,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    landmarks: Optional[Landmarks] = None,
    region_labels: Dict[str, str] = {},
    regions: Optional[PatientRegions] = None,
    show_dose: bool = False,
    study_id: Optional[StudyID] = None,
    **kwargs) -> None:
    pat_ids = arg_to_list(pat_ids, PatientID)

    # Load CT data.
    set = NiftiDataset(dataset)
    plot_ids = []
    ct_datas = []
    spacings = []
    region_datas = []
    landmark_datas = []
    dose_datas = []
    centres = []
    crops = []
    for p in pat_ids:
        pat = set.patient(p)
        if study_id is not None:
            study = pat.study(study_id)
        else:
            study = pat.default_study
        s = study.id
        max_chars = 10
        if len(s) > max_chars:
            s = s[:max_chars]
        plot_id = f"{p}:{s}"
        plot_ids.append(plot_id)
        ct_data = study.ct_data
        ct_datas.append(ct_data)
        spacing = study.ct_spacing
        spacings.append(spacing)

        # Load region data.
        if regions is not None:
            region_data = study.region_data(regions=regions, **kwargs)
        else:
            region_data = None

        # Load landmarks.
        if landmarks is not None:
            landmark_data = study.landmark_data(landmarks=landmarks, use_image_coords=True, **kwargs)
        else:
            landmark_data = None
        landmark_datas.append(landmark_data)

        # Load dose data.
        dose_data = study.dose_data if show_dose else None
        dose_datas.append(dose_data)

        # If 'centre' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
        if centre is not None:
            if isinstance(centre, str):
                if study.has_landmark(centre) and landmark_data is not None and centre not in landmark_data['landmark-id']:
                    centre = study.landmark_data(landmarks=centre)
                elif study.has_regions(centre) and region_data is not None and centre not in region_data:
                    centre = study.region_data(regions=centre)[centre]

        # If 'crop' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
        if crop is not None:
            if isinstance(crop, str):
                if study.has_landmark(crop) and landmark_data is not None and crop not in landmark_data['landmark-id']:
                    crop = study.landmark_data(landmarks=crop)
                elif study.has_regions(crop) and region_data is not None and crop not in region_data:
                    crop = study.region_data(regions=crop)[crop]

        # Apply region labels.
        # This should maybe be moved to base 'plot_patient'? All of the dataset-specific plotting functions
        # use this. Of course 'plot_patient' API would change to include 'region_labels' as an argument.
        region_data, centre, crop = apply_region_labels(region_labels, region_data, centre, crop)
        region_datas.append(region_data)
        centres.append(centre)
        crops.append(crop)

    # Plot.
    okwargs = dict(
        centres=centres,
        crops=crops,
        ct_datas=ct_datas,
        dose_datas=dose_datas,
        landmark_datas=landmark_datas,
        region_datas=region_datas
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
    regions: Optional[PatientRegions] = None,
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

def plot_patient_histogram(
    dataset: str,
    pat_id: PatientID,
    study_id: Optional[StudyID] = None,
    **kwargs) -> None:
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    # Use default study if not 'study_id'.
    ct_data = pat.ct_data if study_id is None else pat.study(study_id).ct_data
    ct_data = ct_data.flatten()
    plot_histogram_base(ct_data, **kwargs)

def plot_registration(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: StudyID,
    fixed_pat_id: PatientID,
    fixed_study_id: StudyID,
    model: str,
    centre: Optional[Union[str, List[str]]] = None,
    crop: Optional[Union[str, List[str]]] = None,
    crop_margin: float = 100,
    idx: Optional[Union[int, float, List[Union[int, float]]]] = None,
    labels: Literal['included', 'excluded', 'all'] = 'all',
    landmarks: Optional[PatientLandmarks] = None,
    regions: Optional[PatientRegions] = None,
    region_labels: Optional[Dict[str, str]] = None,
    transform_format: Literal['itk', 'sitk'] = 'sitk',
    **kwargs) -> None:
    # Find first shared 'centre' and 'crop'.
    # Dealing with shared region labels here - don't worry for now.
    set = NiftiDataset(dataset)
    # centres_of = arg_to_list(centre, str)
    # if centres_of is not None:
    #     for i, c in enumerate(centres_of):
    #         if set.patient(fixed_pat_id).has_regions(c) and set.patient(moving_pat_id).has_regions(c):
    #             centre = c
    #             break
    #         elif i == len(centres_of) - 1:
    #             raise ValueError(f"Could not find shared 'centre' between patients '{fixed_pat_id}' and '{moving_pat_id}'.")
    # crops = arg_to_list(crop, str)
    # if crops is not None and not isinstance(crop, tuple):
    #     for i, c in enumerate(crops):
    #         if set.patient(fixed_pat_id).has_regions(c) and set.patient(moving_pat_id).has_regions(c):
    #             crop = c
    #             break
    #         elif i == len(crops) - 1:
    #             raise ValueError(f"Could not find shared 'crop' between patients '{fixed_pat_id}' and '{moving_pat_id}'.")
    # logging.info(f"Selected 'centre={centre}' and 'crop={crop}'.")

    # Load moving and fixed CT and region data.
    ids = [(moving_pat_id, moving_study_id), (fixed_pat_id, fixed_study_id)]
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
    for i, (p, s) in enumerate(ids):
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
        centre = centres_broad[i]
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
        crop = crops_broad[i]
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
    moved_ct_data, transform, moved_region_data, moved_landmark_data = load_registration(dataset, moving_pat_id, model, fixed_pat_id=fixed_pat_id, fixed_study_id=fixed_study_id, landmarks=landmarks, moving_study_id=moving_study_id, regions=regions, regions_ignore_missing=True, transform_format=transform_format, use_image_coords=True) 

    # Load 'moved_centre' data if not already in 'moved_region_data'.
    centre = centres_broad[2]
    moved_centre = None
    if centre is not None:
        if type(centre) == str:
            if moved_region_data is None or centre not in moved_region_data:
                _, _, centre_region_data = load_registration(dataset, moving_pat_id, model, fixed_pat_id=fixed_pat_id, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id, regions=centre, transform_format=transform_format) 
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
                _, _, crop_region_data = load_registration(dataset, moving_pat_id, model, fixed_pat_id=fixed_pat_id, fixed_study_id=fixed_study_id, moving_study_id=moving_study_id, regions=crop, transform_format=transform_format) 
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

    # Plot.
    moving_ct_data, fixed_ct_data = ct_datas
    moving_centre, fixed_centre = centres
    moving_crop, fixed_crop = crops
    moving_spacing, fixed_spacing = spacings
    moving_offset, fixed_offset = offsets
    moving_landmark_data, fixed_landmark_data = landmark_datas
    moving_region_data, fixed_region_data = region_datas
    moving_idx, fixed_idx, moved_idx = idxs_broad
    okwargs = dict(
        fixed_centre=fixed_centre,
        fixed_crop=fixed_crop,
        fixed_crop_margin=crop_margin,
        fixed_ct_data=fixed_ct_data,
        fixed_idx=fixed_idx,
        fixed_landmark_data=fixed_landmark_data,
        fixed_offset=fixed_offset,
        fixed_spacing=fixed_spacing,
        fixed_region_data=fixed_region_data,
        moved_centre=moved_centre,
        moved_crop=moved_crop,
        moved_crop_margin=crop_margin,
        moved_ct_data=moved_ct_data,
        moved_idx=moved_idx,
        moved_landmark_data=moved_landmark_data,
        moved_offset=fixed_offset,
        moved_region_data=moved_region_data,
        moving_centre=moving_centre,
        moving_crop=moving_crop,
        moving_crop_margin=crop_margin,
        moving_ct_data=moving_ct_data,
        moving_idx=moving_idx,
        moving_landmark_data=moving_landmark_data,
        moving_offset=moving_offset,
        moving_spacing=moving_spacing,
        moving_region_data=moving_region_data,
        transform=transform,
        transform_format=transform_format,
    )
    plot_registration_base(moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, **okwargs, **kwargs)

def plot_segmenter_predictions(
    dataset: str,
    pat_id: str,
    model: str,
    centre: Optional[str] = None,
    crop: Optional[str] = None,
    regions: PatientRegions = 'all',
    regions_model: PatientRegions = 'all',
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

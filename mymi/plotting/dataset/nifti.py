from collections.abc import Iterable
import numpy as np
import re
from typing import *

from mymi.dataset import NiftiDataset
from mymi.gradcam.dataset.nifti import load_multi_segmenter_heatmap
from mymi import logging
from mymi.prediction.dataset.nifti import create_localiser_prediction, create_adaptive_segmenter_prediction, create_multi_segmenter_prediction, create_segmenter_prediction, get_localiser_prediction, load_localiser_centre, load_localiser_prediction, load_registration, load_segmenter_prediction, load_adaptive_segmenter_prediction, load_multi_segmenter_prediction, load_multi_segmenter_prediction_dict
from mymi.regions import regions_to_list
from mymi.types import *
from mymi.utils import arg_broadcast, arg_to_list

from ..plotting import apply_region_labels
from ..plotting import plot_heatmap as plot_heatmap_base
from ..plotting import plot_histogram as plot_histogram_base
from ..plotting import plot_localiser_prediction as plot_localiser_prediction_base
from ..plotting import plot_multi_segmenter_prediction as plot_multi_segmenter_prediction_base
from ..plotting import plot_segmenter_prediction as plot_segmenter_prediction_base
from ..plotting import plot_segmenter_prediction_diff as plot_segmenter_prediction_diff_base
from ..plotting import plot_patient as plot_patient_base
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

def plot_patient(
    dataset: str,
    pat_id: str,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    landmarks: Optional[Landmarks] = None,
    region_labels: Dict[str, str] = {},
    regions: Optional[PatientRegions] = None,
    show_dose: bool = False,
    study_id: Optional[StudyID] = None,
    **kwargs) -> None:

    # Load CT data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    if study_id is not None:
        study = pat.study(study_id)
    else:
        study = pat.default_study
    ct_data = study.ct_data
    spacing = study.ct_spacing

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

    # Load dose data.
    dose_data = study.dose_data if show_dose else None

    # If 'centre' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
    if centre is not None:
        if isinstance(centre, str):
            if study.has_landmark(centre) and landmark_data is not None and centre not in landmark_data['landmark-id']:
                centre = study.landmark_data(landmarks=centre)
            elif study.has_region(centre) and region_data is not None and centre not in region_data:
                centre = study.region_data(regions=centre)[centre]

    # If 'crop' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
    if crop is not None:
        if isinstance(crop, str):
            if study.has_landmark(crop) and landmark_data is not None and crop not in landmark_data['landmark-id']:
                crop = study.landmark_data(landmarks=crop)
            elif study.has_region(crop) and region_data is not None and crop not in region_data:
                crop = study.region_data(regions=crop)[crop]

    # Apply region labels.
    # This should maybe be moved to base 'plot_patient'? All of the dataset-specific plotting functions
    # use this. Of course 'plot_patient' API would change to include 'region_labels' as an argument.
    region_data, centre, crop = apply_region_labels(region_labels, region_data, centre, crop)

    # Plot.
    max_chars = 10
    study_id = study.id
    if len(study_id) > max_chars:
        study_id = study_id[:max_chars]
    id = f"{pat_id}:{study_id}"
    plot_patient_base(id, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, dose_data=dose_data, landmark_data=landmark_data, region_data=region_data, **kwargs)

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
    plot_patient_base(plot_id, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, dose_data=dose_data, region_data=region_data, **kwargs)

def plot_patient_histogram(
    dataset: str,
    pat_id: PatientID,
    study_id: Optional[StudyID] = None,
    **kwargs) -> None:
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    # Use default study if not 'study_id'.
    ct_data = pat.ct_data if study_id is None else pat.study(study_id).ct_data
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
    #         if set.patient(fixed_pat_id).has_region(c) and set.patient(moving_pat_id).has_region(c):
    #             centre = c
    #             break
    #         elif i == len(centres_of) - 1:
    #             raise ValueError(f"Could not find shared 'centre' between patients '{fixed_pat_id}' and '{moving_pat_id}'.")
    # crops = arg_to_list(crop, str)
    # if crops is not None and not isinstance(crop, tuple):
    #     for i, c in enumerate(crops):
    #         if set.patient(fixed_pat_id).has_region(c) and set.patient(moving_pat_id).has_region(c):
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
    moved_ct_data, transform, moved_region_data, moved_landmark_data = load_registration(dataset, moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, landmarks=landmarks, regions=regions, regions_ignore_missing=True, transform_format=transform_format, use_image_coords=True) 

    # Load 'moved_centre' data if not already in 'moved_region_data'.
    centre = centres_broad[2]
    moved_centre = None
    if centre is not None:
        if type(centre) == str:
            if moved_region_data is None or centre not in moved_region_data:
                _, _, centre_region_data = load_registration(dataset, moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, regions=centre, transform_format=transform_format) 
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
                _, _, crop_region_data = load_registration(dataset, moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, regions=crop, transform_format=transform_format) 
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

def plot_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: ModelName,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    load_prediction: bool = True,
    region: Optional[PatientRegions] = None,
    region_label: Optional[Dict[str, str]] = None,
    show_ct: bool = True,
    **kwargs) -> None:
    regions = arg_to_list(region, str)
    region_labels = arg_to_list(region_label, str)
    
    # Load data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data if show_ct else None
    region_data = pat.region_data(region=regions) if regions is not None else None
    spacing = pat.ct_spacing

    # Load prediction.
    if load_prediction:
        pred = load_localiser_prediction(dataset, pat_id, localiser)
    else:
        # Make prediction.
        pred = get_localiser_prediction(dataset, pat_id, localiser)

    if centre is not None:
        if type(centre) == str:
            if region_data is None or centre not in region_data:
                centre = pat.region_data(region=centre)[centre]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = pat.region_data(region=crop)[crop]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre' and 'crop' keys.
        if type(centre) == str and centre in region_labels:
            centre = region_labels[centre] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]
    
    # Plot.
    pred_region = localiser[0].split('-')[1]    # Infer pred region name from localiser model name.
    plot_localiser_prediction_base(pat_id, spacing, pred, pred_region, centre=centre, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

def plot_adaptive_segmenter_prediction(
    dataset: str,
    pat_id: str,
    model: Union[ModelName, List[ModelName]],
    model_regions: PatientRegions,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    load_pred: bool = True,
    model_label: Union[str, List[str]] = None,
    model_spacing: Optional[ImageSpacing3D] = None,
    n_epochs: Optional[int] = None,
    pred_label: Union[str, List[str]] = None,
    pred_regions: Optional[Union[str, List[str]]] = None,
    region: Optional[Union[str, List[str]]] = None,
    region_label: Optional[Union[str, List[str]]] = None,
    seg_spacings: Optional[Union[ImageSpacing3D, List[ImageSpacing3D]]] = (1, 1, 2),
    show_ct: bool = True,
    **kwargs) -> None:
    models = arg_to_list(model, tuple)
    # If only a single model, allow 'model_region=Brain' or 'model_region=['Brain']'.
    # If multiple models, list of lists must be specified, e.g. 'model_region=[['Brain'], 'Brainstem']'.
    #   Flat list not supported, e.g. 'model_region=['Brain', 'Brainstem']'.
    if len(models) == 1:
        model_regionses = [regions_to_list(model_regions)]
    else:
        model_regionses = model_regions
    regions = regions_to_list(region)
    region_labels = arg_to_list(region_label, str)
    if region_labels is not None:
        assert len(regions) == len(region_labels)
        region_label_map = dict(zip(regions, region_labels))
    else:
        region_label_map = None
    pred_regions = regions_to_list(pred_regions)
    n_models = len(models)
    model_labels = arg_to_list(model_label, str)
    if model_labels is None:
        model_labels = list(f'model:{i}' for i in range(n_models))

    # Infer 'pred_regions' from localiser model names.
    if type(seg_spacings) == tuple:
        seg_spacings = [seg_spacings] * n_models
    else:
        assert len(seg_spacings) == n_models
    
    # Load data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data if show_ct else None
    region_data = pat.region_data(region=regions) if regions is not None else None
    spacing = pat.ct_spacing

    # Load predictions.
    pred_data = {}
    for i in range(n_models):
        model = models[i]
        model_regions = model_regionses[i]
        model_label = model_labels[i]

        # Load segmenter prediction.
        pred = None
        if load_pred:
            pred_exists = load_adaptive_segmenter_prediction(dataset, pat_id, model, exists_only=True)
            if not pred_exists:
                if model_spacing is None or n_epochs is None:
                    # Raise error on purpose to get full message.
                    try:
                        load_adaptive_segmenter_prediction(dataset, pat_id, model, exists_only=False)
                    except ValueError as e:
                        if model_spacing is None:
                            logging.error(f"Model prediction doesn't exist, so 'model_spacing' is required to make prediction.")
                        if n_epochs is None:
                            logging.error(f"Model prediction doesn't exist, so 'n_epochs' is required to make prediction.")
                        raise e
                logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', model '{model}'.")
                create_adaptive_segmenter_prediction(dataset, pat_id, model, model_regions, model_spacing, n_epochs=n_epochs)

            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', model '{model}'.")
            pred = load_adaptive_segmenter_prediction(dataset, pat_id, model)

        # Split multi-channel prediction by region.
        n_regions = len(model_regions)
        if pred.shape[0] != n_regions + 1:
            raise ValueError(f"With 'model_regions={model_regions}', expected {n_regions + 1} channels in prediction for dataset '{dataset}', patient '{pat_id}', model '{model}', got {pred.shape[0]}.")
        for r, p_data in zip(model_regions, pred[1:]):
            if pred_regions is not None and r not in pred_regions:
                continue
            pred_data[f'{model_label}:{r}'] = p_data

    if centre is not None:
        match = re.search(MODEL_SELECT_PATTERN_MULTI, centre)
        if match is not None:
            if match.group(2) is None:
                assert n_models == 1
                model_i = 0
            else:
                model_i = int(match.group(2))
                assert model_i < n_models
            region = match.group(3)
            p_label = f'model:{model_i}:{region}'
            centre_tmp = centre
            print(pred_data.keys())
            centre = pred_data[p_label]
            if centre.sum() == 0:
                raise ValueError(f"Got empty prediction for 'centre={centre_tmp}, please provide 'idx' instead.")
        elif region_data is None or centre not in region_data:
            centre = pat.region_data(region=centre)[centre]

    if type(crop) == str:
        match = re.search(MODEL_SELECT_PATTERN_MULTI, crop)
        if match is not None:
            if match.group(2) is None:
                assert n_models == 1
                model_i = 0
            else:
                model_i = int(match.group(2))
                assert model_i < n_models
            region = match.group(3)
            p_label = f'model:{model_i}:{region}'
            crop_tmp = crop
            crop = pred_data[p_label]
            if crop.sum() == 0:
                raise ValueError(f"Got empty prediction for 'crop={crop_tmp}, please provide alternative 'crop'.")
        elif region_data is None or crop not in region_data:
            crop = pat.region_data(region=crop)[crop]

    # Rename regions.
    # Also need to rename predicted regions.
    if region_label_map is not None:
        for old_region, new_region in region_label_map.items():
            region_data[new_region] = region_data.pop(old_region)

            # Rename 'centre' and 'crop' keys.
            if type(centre) == str and centre == old_region:
                centre = new_region
            if type(crop) == str and crop == old_region:
                crop = new_region

            # Rename predicted regions.
            for i in range(n_models):
                model_label = model_labels[i]
                pred_data[f'{new_region} ({model_label})'] = pred_data.pop(f'{model_label}:{old_region}')
    
    # Plot.
    plot_multi_segmenter_prediction_base(pat_id, spacing, pred_data, centre=centre, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

def plot_multi_segmenter_prediction(
    dataset: str,
    pat_id: str,
    model: Union[ModelName, List[ModelName]],
    model_region: PatientRegions,
    centre: Optional[str] = None,
    check_epochs: bool = True,
    crop: Optional[Union[str, Box2D]] = None,
    load_pred: bool = True,
    model_label: Union[str, List[str]] = None,
    model_spacing: Optional[ImageSpacing3D] = None,
    n_epochs: Optional[int] = None,
    pred_region: Optional[PatientRegions] = None,
    region: Optional[PatientRegions] = None,
    region_label: Optional[Union[str, List[str]]] = None,
    seg_spacings: Optional[Union[ImageSpacing3D, List[ImageSpacing3D]]] = (1, 1, 2),
    show_ct: bool = True,
    **kwargs) -> None:
    models = arg_to_list(model, tuple)
    # If only a single model, allow 'model_region=Brain' or 'model_region=['Brain']'.
    # If multiple models, list of lists must be specified, e.g. 'model_region=[['Brain'], 'Brainstem']'.
    #   Flat list not supported, e.g. 'model_region=['Brain', 'Brainstem']'.
    if len(models) == 1:
        model_regionses = [regions_to_list(model_region)]
    else:
        model_regionses = model_region
    regions = regions_to_list(region)
    region_labels = arg_to_list(region_label, str)
    if region_labels is not None:
        assert len(regions) == len(region_labels)
        region_label_map = dict(zip(regions, region_labels))
    else:
        region_label_map = None
    n_models = len(models)
    model_labels = arg_to_list(model_label, str)
    if model_labels is None:
        model_labels = list(f'model:{i}' for i in range(n_models))
    pred_regions = regions_to_list(pred_region)

    # Infer 'pred_regions' from localiser model names.
    if type(seg_spacings) == tuple:
        seg_spacings = [seg_spacings] * n_models
    else:
        assert len(seg_spacings) == n_models
    
    # Load data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data if show_ct else None
    region_data = pat.region_data(region=regions) if regions is not None else None
    spacing = pat.ct_spacing

    # Load predictions.
    pred_data = {}
    for i in range(n_models):
        model = models[i]
        model_regions = model_regionses[i]
        model_label = model_labels[i]

        # Load segmenter prediction.
        pred = None
        if load_pred:
            pred_exists = load_multi_segmenter_prediction(dataset, pat_id, model, exists_only=True)
            if not pred_exists:
                if model_spacing is None or (check_epochs and n_epochs is None):
                    try:
                        # Call method again to get full error message.
                        load_multi_segmenter_prediction(dataset, pat_id, model, exists_only=False)
                    except ValueError as e:
                        if model_spacing is None:
                            logging.error(f"Making model prediction - must pass 'model_spacing'.")
                        if check_epochs and n_epochs is None:
                            logging.error(f"Making model prediction with 'check_epochs=True' - must pass 'n_epochs'.")
                        raise e
                logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', model '{model}'.")
                create_multi_segmenter_prediction(dataset, pat_id, model, model_regions, model_spacing, check_epochs=check_epochs, n_epochs=n_epochs, **kwargs)

            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', model '{model}'.")
            pred = load_multi_segmenter_prediction(dataset, pat_id, model)

        # Split the prediction into regions by channel.
        n_regions = len(model_regions)
        if pred.shape[0] != n_regions + 1:
            raise ValueError(f"With 'model_regions={model_regions}', expected {n_regions + 1} channels in prediction for dataset '{dataset}', patient '{pat_id}', model '{model}', got {pred.shape[0]}.")
        for r, p_data in zip(model_regions, pred[1:]):
            if pred_regions is not None and r not in pred_regions:
                continue
            pred_data[f'{model_label}:{r}'] = p_data

    if centre is not None:
        match = re.search(MODEL_SELECT_PATTERN_MULTI, centre)
        if match is not None:
            if match.group(2) is None:
                assert n_models == 1
                model_i = 0
            else:
                model_i = int(match.group(2))
                assert model_i < n_models
            region = match.group(3)
            model_label = model_labels[model_i]
            label = f'{model_label}:{region}'
            centre_tmp = centre
            if label not in pred_data:
                raise ValueError(f"Requested 'centre={centre_tmp}' not found in prediction data.")
            centre = pred_data[label]
            if centre.sum() == 0:
                raise ValueError(f"Got empty prediction for 'centre={centre_tmp}, please provide 'idx' instead.")
        elif region_data is None or centre not in region_data:
            centre = pat.region_data(region=centre)[centre]

    if type(crop) == str:
        match = re.search(MODEL_SELECT_PATTERN_MULTI, crop)
        if match is not None:
            if match.group(2) is None:
                assert n_models == 1
                model_i = 0
            else:
                model_i = int(match.group(2))
                assert model_i < n_models
            region = match.group(3)
            model_label = model_labels[model_i]
            label = f'{model_label}:{region}'
            crop_tmp = crop
            if label not in pred_data:
                raise ValueError(f"Requested 'crop={crop_tmp}' not found in prediction data.")
            crop = pred_data[label]
            if crop.sum() == 0:
                raise ValueError(f"Got empty prediction for 'crop={crop_tmp}, please provide alternative 'crop'.")
        elif region_data is None or crop not in region_data:
            crop = pat.region_data(region=crop)[crop]

    if region_labels is not None:
        for old, new in zip(regions, region_labels):
            # Rename 'region_data' keys.
            region_data[new] = region_data.pop(old)

            # Rename 'centre' and 'crop' keys.
            if type(centre) == str and centre == old:
                centre = new
            if type(crop) == str and crop == old:
                crop = new

            for i in range(n_models):
                model_label = model_labels[i]
                pred_data[f'{new} ({model_label})'] = pred_data.pop(f'{model_label}:{old}')
    
    # Plot.
    plot_multi_segmenter_prediction_base(pat_id, spacing, pred_data, centre=centre, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

def plot_segmenter_prediction(
    dataset: str,
    pat_id: str,
    localiser: Union[ModelName, List[ModelName]],
    segmenter: Union[ModelName, List[ModelName]],
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    load_loc_pred: bool = True,
    load_seg_pred: bool = True,
    pred_labels: Dict[str, str] = {},
    region_labels: Dict[str, str] = {},
    regions: Optional[PatientRegions] = None,
    show_ct: bool = True,
    seg_spacings: Optional[Union[ImageSpacing3D, List[ImageSpacing3D]]] = (1, 1, 2),
    **kwargs) -> None:
    localisers = arg_to_list(localiser, tuple)
    segmenters = arg_to_list(segmenter, tuple)
    regions = regions_to_list(regions)
    localisers = arg_broadcast(localisers, segmenters)
    n_models = len(localisers)

    # Infer 'pred_regions' from localiser model names.
    if type(seg_spacings) == tuple:
        seg_spacings = [seg_spacings] * n_models
    else:
        assert len(seg_spacings) == n_models
    
    # Load data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data if show_ct else None
    spacing = pat.ct_spacing
    if regions is not None:
        region_data = pat.region_data(regions=regions)
    else:
        region_data = None

    # Set prediction labels.
    for i in range(n_models):
        def_label = f'model:{i}'
        if def_label not in pred_labels:
            pred_labels[def_label] = def_label

    # Load predictions.
    loc_centres = []
    pred_data = {}
    for i in range(n_models):
        localiser = localisers[i]
        segmenter = segmenters[i]
        pred_label = pred_labels[f'model:{i}']

        # Load/make localiser prediction.
        if load_loc_pred:
            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")
            try:
                loc_centre = load_localiser_centre(dataset, pat_id, localiser)
            except ValueError as e:
                loc_centre = None
                logging.info(f"No prediction found for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")

        if loc_centre is None:
            logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")
            create_localiser_prediction(dataset, pat_id, localiser)
            loc_centre = load_localiser_centre(dataset, pat_id, localiser)

        # Get segmenter prediction.
        pred = None
        # Attempt load.
        if load_seg_pred:
            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}'...")
            try:
                pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)
            except ValueError as e:
                logging.info(str(e))
        # Make prediction if didn't/couldn't load.
        if pred is None:
            logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}'...")
            create_segmenter_prediction(dataset, pat_id, localiser, segmenter)           # Handle multiple spacings.
            pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)

        loc_centres.append(loc_centre)
        pred_data[pred_label] = pred

    if centre is not None:
        if centre == 'model':
            assert n_models == 1
            centre = pred_data[pred_labels[0]]
        elif type(centre) == str:
            match = re.search(MODEL_SELECT_PATTERN, centre)
            if match is not None:
                model_i = int(match.group(1))
                assert model_i < n_models
                centre = pred_data[pred_labels[f'model:{model_i}']]
            elif region_data is None or centre not in region_data:
                centre = pat.region_data(region=centre)[centre]

    if type(crop) == str:
        if crop == 'model':
            assert n_models == 1
            crop = pred_data[pred_label[0]]
        else:
            match = re.search(MODEL_SELECT_PATTERN, crop)
            if match is not None:
                model_i = int(match.group(1))
                assert model_i < n_models
                crop = pred_data[pred_label[model_i]]
            elif region_data is None or crop not in region_data:
                crop = pat.region_data(region=crop)[crop]

    # Apply region labels.
    region_data, centre, crop = apply_region_labels(region_labels, region_data, centre, crop)
    
    # Plot.
    plot_segmenter_prediction_base(pat_id, spacing, pred_data, centre=centre, crop=crop, ct_data=ct_data, loc_centre=loc_centres, region_data=region_data, **kwargs)

def plot_segmenter_prediction_diff(
    dataset: str,
    pat_id: str,
    localiser: Union[ModelName, List[ModelName]],
    segmenter: Union[ModelName, List[ModelName]],
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    load_loc_pred: bool = True,
    load_seg_pred: bool = True,
    diff_label: Union[str, List[str]] = None,
    show_ct: bool = True,
    **kwargs) -> None:
    localisers = arg_to_list(localiser, tuple)
    segmenters = arg_to_list(segmenter, tuple)
    localisers = arg_broadcast(localisers, segmenters)
    n_models = len(localisers)
    diff_labels = arg_to_list(diff_label, str)

    # Infer 'diff_regions' from localiser model names.
    diff_regions = [l[0].split('-')[1] for l in localisers]
    
    # Load data.
    set = NiftiDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data if show_ct else None
    spacing = pat.ct_spacing

    # Load pred/region data.
    pred_datas = []
    region_datas = []
    for i in range(n_models):
        localiser = localisers[i]
        segmenter = segmenters[i]
        diff_region = diff_regions[i]
        region_data = pat.region_data(region=diff_region)[diff_region]
        region_datas.append(region_data)

        # Load/make localiser prediction.
        if load_loc_pred:
            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")
            try:
                loc_centre = load_localiser_centre(dataset, pat_id, localiser)
            except ValueError as e:
                loc_centre = None
                logging.info(f"No prediction found for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")

        if loc_centre is None:
            logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}'...")
            create_localiser_prediction(dataset, pat_id, localiser)
            loc_centre = load_localiser_centre(dataset, pat_id, localiser)

        # Get segmenter prediction.
        pred = None
        # Attempt load.
        if load_seg_pred:
            logging.info(f"Loading prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}'...")
            try:
                pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)
            except ValueError as e:
                logging.info(str(e))
        # Make prediction if didn't/couldn't load.
        if pred is None:
            logging.info(f"Making prediction for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}'...")
            create_segmenter_prediction(dataset, pat_id, localiser, segmenter)           # Handle multiple spacings.
            pred = load_segmenter_prediction(dataset, pat_id, localiser, segmenter)

        pred_datas.append(pred)

    # Reduce region diffs - can take a while.
    pred_data = np.stack(pred_datas, axis=0).astype(int)
    region_data = np.stack(region_datas, axis=0).astype(int)
    diff_data = pred_data - region_data
    diff_data = diff_data.reshape(n_models, -1)
    diff_data = np.apply_along_axis(__reduce_region_diffs, 0, diff_data)
    diff_data = diff_data.reshape(ct_data.shape)

    # Create plottable masks.
    if diff_labels is None:
        diff_labels = ['pred-only', 'region-only']
    else:
        assert len(diff_labels) == 2
    pred_only_data = np.zeros(ct_data.shape, dtype=bool)
    pred_only_data[np.where(diff_data == 1)] = True
    region_only_data = np.zeros(ct_data.shape, dtype=bool)
    region_only_data[np.where(diff_data == -1)] = True
    diff_data = {
        diff_labels[0]: pred_only_data, 
        diff_labels[1]: region_only_data
    }
    
    # Plot.
    plot_segmenter_prediction_diff_base(pat_id, spacing, diff_data, centre=centre, crop=crop, ct_data=ct_data, **kwargs)

def __reduce_region_diffs(diffs: List[int]) -> int:
    n_pos = 0
    n_neg = 0
    for diff in diffs:
        if diff == -1:
            n_neg += 1
        elif diff == 1:
            n_pos += 1
    if n_pos == 0:
        if n_neg >= 1:
            return -1 # If one or more regions have neg diffs, show neg diff.
    elif n_neg == 0:
        if n_pos >= 1:
            return 1 # If one or more regions have pos diffs, show pos diff.
        
    # If no pos/neg diffs, or conflicting diffs, show nothing.
    return 0

from collections.abc import Iterable
import numpy as np
import re
from typing import *
from wand import image

from mymi.datasets.nifti import NiftiDataset, Modality
from mymi.gradcam.dataset.nifti import load_multi_segmenter_heatmap
from mymi.predictions.datasets.nifti import load_registration, load_segmenter_predictions
from mymi.typing import *
from mymi.utils import *

from ..plotting import plot_dataset_histogram as pdh, plot_patient_histograms as pph, plot_loaded, plot_patients as pp, plot_registrations as plot_registrations_base, plot_segmenter_predictions as plot_segmenter_predictions_base

MODEL_SELECT_PATTERN = r'^model:([0-9]+)$'
MODEL_SELECT_PATTERN_MULTI = r'^model(:([0-9]+))?:([a-zA-Z_]+)$'

def plot_dataset_histogram(*args, **kwargs) -> None:
    pdh(NiftiDataset, *args, **kwargs)

def plot_patients(*args, **kwargs) -> None:
    pp(NiftiDataset, *args, **kwargs)

def plot_patient_histograms(*args, **kwargs) -> None:
    pph(NiftiDataset, *args, **kwargs)

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

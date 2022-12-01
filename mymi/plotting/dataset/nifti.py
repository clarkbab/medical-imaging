import numpy as np
import re
from typing import Dict, List, Optional, Union

from mymi import dataset as ds
from mymi import logging
from mymi.prediction.dataset.nifti import create_localiser_prediction, create_segmenter_prediction, get_localiser_prediction, load_localiser_centre, load_localiser_prediction, load_segmenter_prediction
from mymi import types
from mymi.utils import arg_broadcast, arg_to_list

from ..plotter import plot_localiser_prediction as plot_localiser_prediction_base
from ..plotter import plot_segmenter_prediction as plot_segmenter_prediction_base
from ..plotter import plot_segmenter_prediction_diff as plot_segmenter_prediction_diff_base
from ..plotter import plot_region as plot_region_base

MODEL_SELECT_PATTERN = r'^model:([0-9]+)$'

def plot_region(
    dataset: str,
    pat_id: str,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    region: Optional[types.PatientRegions] = None,
    region_label: Optional[Dict[str, str]] = None,     # Gives 'regions' different names to those used for loading the data.
    show_dose: bool = False,
    **kwargs) -> None:
    regions = arg_to_list(region, str)
    region_labels = arg_to_list(region_label, str)

    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=regions) if regions is not None else None
    spacing = patient.ct_spacing
    dose_data = patient.dose_data if show_dose else None

    if centre_of is not None:
        if type(centre_of) == str:
            if region_data is None or centre_of not in region_data:
                centre_of = patient.region_data(regions=centre_of)[centre_of]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = patient.region_data(regions=crop)[crop]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre_of' and 'crop' keys.
        if type(centre_of) == str and centre_of in region_labels:
            centre_of = region_labels[centre_of] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]

    # Plot.
    plot_region_base(pat_id, ct_data.shape, spacing, centre_of=centre_of, crop=crop, ct_data=ct_data, dose_data=dose_data, region_data=region_data, **kwargs)

def plot_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.ModelName,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    load_prediction: bool = True,
    region: Optional[types.PatientRegions] = None,
    region_label: Optional[Dict[str, str]] = None,
    show_ct: bool = True,
    **kwargs) -> None:
    regions = arg_to_list(region, str)
    region_labels = arg_to_list(region_label, str)
    
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    region_data = patient.region_data(regions=regions) if regions is not None else None
    spacing = patient.ct_spacing

    # Load prediction.
    if load_prediction:
        pred = load_localiser_prediction(dataset, pat_id, localiser)
    else:
        # Make prediction.
        pred = get_localiser_prediction(dataset, pat_id, localiser)

    if centre_of is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                centre_of = patient.region_data(regions=centre_of)[centre_of]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = patient.region_data(regions=crop)[crop]

    if region_labels is not None:
        # Rename 'regions' and 'region_data' keys.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        for old, new in region_labels.items():
            region_data[new] = region_data.pop(old)

        # Rename 'centre_of' and 'crop' keys.
        if type(centre_of) == str and centre_of in region_labels:
            centre_of = region_labels[centre_of] 
        if type(crop) == str and crop in region_labels:
            crop = region_labels[crop]
    
    # Plot.
    pred_region = localiser[0].split('-')[1]    # Infer pred region name from localiser model name.
    plot_localiser_prediction_base(pat_id, spacing, pred, pred_region, centre_of=centre_of, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

def plot_segmenter_prediction(
    dataset: str,
    pat_id: str,
    localiser: Union[types.ModelName, List[types.ModelName]],
    segmenter: Union[types.ModelName, List[types.ModelName]],
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    load_loc_pred: bool = True,
    load_seg_pred: bool = True,
    pred_label: Union[str, List[str]] = None,
    region: Optional[Union[str, List[str]]] = None,
    region_label: Optional[Union[str, List[str]]] = None,
    show_ct: bool = True,
    seg_spacings: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (1, 1, 2),
    **kwargs) -> None:
    localisers = arg_to_list(localiser, tuple)
    segmenters = arg_to_list(segmenter, tuple)
    regions = arg_to_list(region, str) if region is not None else None
    region_labels = arg_to_list(region_label, str) if region_label is not None else None
    localisers = arg_broadcast(localisers, segmenters)
    n_models = len(localisers)
    if pred_label is not None:
        pred_labels = arg_to_list(pred_label, str)
    else:
        pred_labels = list(f'model-{i}' for i in range(n_models))

    # Infer 'pred_regions' from localiser model names.
    if type(seg_spacings) == tuple:
        seg_spacings = [seg_spacings] * n_models
    else:
        assert len(seg_spacings) == n_models
    
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    region_data = patient.region_data(regions=regions) if regions is not None else None
    spacing = patient.ct_spacing

    # Load predictions.
    loc_centres = []
    pred_data = {}
    for i in range(n_models):
        localiser = localisers[i]
        segmenter = segmenters[i]
        pred_label = pred_labels[i]

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

    if centre_of is not None:
        if centre_of == 'model':
            assert n_models == 1
            centre_of = pred_data[pred_labels[0]]
        elif type(centre_of) == str:
            match = re.search(MODEL_SELECT_PATTERN, centre_of)
            if match is not None:
                model_i = int(match.group(1))
                assert model_i < n_models
                centre_of = pred_data[pred_labels[model_i]]
            elif region_data is None or centre_of not in region_data:
                centre_of = patient.region_data(regions=centre_of)[centre_of]

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
                crop = patient.region_data(regions=crop)[crop]

    if region_labels is not None:
        for old, new in zip(regions, region_labels):
            # Rename 'region_data' keys.
            region_data[new] = region_data.pop(old)

            # Rename 'centre_of' and 'crop' keys.
            if type(centre_of) == str and centre_of == old:
                centre_of = new
            if type(crop) == str and crop == old:
                crop = new

        # Rename 'regions'.
        regions = region_labels
    
    # Plot.
    plot_segmenter_prediction_base(pat_id, spacing, pred_data, centre_of=centre_of, crop=crop, ct_data=ct_data, loc_centre=loc_centres, region_data=region_data, **kwargs)

def plot_segmenter_prediction_diff(
    dataset: str,
    pat_id: str,
    localiser: Union[types.ModelName, List[types.ModelName]],
    segmenter: Union[types.ModelName, List[types.ModelName]],
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
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
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    spacing = patient.ct_spacing

    # Load pred/region data.
    pred_datas = []
    region_datas = []
    for i in range(n_models):
        localiser = localisers[i]
        segmenter = segmenters[i]
        diff_region = diff_regions[i]
        region_data = patient.region_data(regions=diff_region)[diff_region]
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
    plot_segmenter_prediction_diff_base(pat_id, spacing, diff_data, centre_of=centre_of, crop=crop, ct_data=ct_data, **kwargs)

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

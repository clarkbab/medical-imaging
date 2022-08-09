import re
from typing import Dict, List, Optional, Union

from mymi import dataset as ds
from mymi.geometry import get_extent_centre
from mymi import logging
from mymi.prediction.dataset.nifti import create_patient_localiser_prediction, create_patient_segmenter_prediction, load_patient_localiser_centre, load_patient_localiser_prediction, load_patient_segmenter_prediction
from mymi import types

from ..plotter import plot_localiser_prediction, plot_regions, plot_segmenter_prediction

PRED_PATTERN = r'^pred:([0-9]+)$'

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    regions: Optional[types.PatientRegions] = None,
    region_labels: Optional[Dict[str, str]] = None,
    show_dose: bool = False,
    **kwargs) -> None:
    if type(regions) == str:
        regions = [regions]
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=regions) if regions is not None else None
    spacing = patient.ct_spacing
    dose_data = patient.dose_data if show_dose else None

    # Add 'centre_of' region data.
    if centre_of is not None:
        if region_data is None:
            centre_of_data = patient.region_data(regions=centre_of)
            region_data = centre_of_data
        elif centre_of not in region_data.keys():
            centre_of_data = patient.region_data(regions=centre_of)
            region_data[centre_of] = centre_of_data[centre_of]

    # Add 'crop' region data.
    if crop is not None:
        if type(crop) == str:
            if region_data is None:
                crop_data = patient.region_data(regions=crop)
                region_data = crop_data
            elif crop not in region_data.keys():
                crop_data = patient.region_data(regions=crop)
                region_data[crop] = crop_data[crop]

    # Relabel regions.
    if region_labels is not None:
        # Names.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        # Region data.
        for old_label, new_label in region_labels.items():
            region_data[new_label] = region_data.pop(old_label)
        # Args.
        if centre_of is not None:
            centre_of = region_labels[centre_of]
        if crop is not None:
            if type(crop) == str:
                crop = region_labels[crop]

    # Plot.
    plot_regions(pat_id, ct_data.shape, spacing, centre_of=centre_of, crop=crop, ct_data=ct_data, dose_data=dose_data, regions=regions, region_data=region_data, **kwargs)

def plot_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    localiser: types.ModelName,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    load_prediction: bool = True,
    regions: Optional[types.PatientRegions] = None,
    show_ct: bool = True,
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    region_data = patient.region_data(regions=regions) if regions is not None else None
    spacing = patient.ct_spacing

    # Load prediction.
    if load_prediction:
        pred = load_patient_localiser_prediction(dataset, pat_id, localiser)
    else:
        # Set truncation if 'SpinalCord'.
        truncate = True if 'SpinalCord' in localiser[0] else False

        # Make prediction.
        pred = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, truncate=truncate)

    # Add 'centre_of' region data.
    if centre_of is not None:
        if region_data is None:
            centre_of_data = patient.region_data(regions=centre_of)
            region_data = centre_of_data
        elif centre_of not in region_data.keys():
            centre_of_data = patient.region_data(regions=centre_of)
            region_data[centre_of] = centre_of_data[centre_of]

    # Add 'crop' region data.
    if type(crop) == str:
        if region_data is None:
            crop_data = patient.region_data(regions=crop)
            region_data = crop_data
        elif crop not in region_data.keys():
            crop_data = patient.region_data(regions=crop)
            region_data[crop] = crop_data[crop]
    
    # Plot.
    plot_localiser_prediction(pat_id, spacing, pred, centre_of=centre_of, crop=crop, ct_data=ct_data, regions=regions, region_data=region_data, **kwargs)

def plot_patient_segmenter_prediction(
    dataset: str,
    pat_id: str,
    localisers: Union[types.ModelName, List[types.ModelName]],
    segmenters: Union[types.ModelName, List[types.ModelName]],
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    loc_sizes: Optional[Union[types.ImageSize3D, List[types.ImageSize3D]]] = (128, 128, 150),
    loc_spacings: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (4, 4, 4),
    pred_regions: Optional[Union[str, List[str]]] = None,
    regions: Optional[types.PatientRegions] = None,
    region_labels: Optional[Dict[str, str]] = None,
    show_ct: bool = True,
    seg_spacings: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (1, 1, 2),
    **kwargs) -> None:
    # Convert args to list.
    if type(localisers) == tuple:
        localisers = [localisers]
    if type(segmenters) == tuple:
        segmenters = [segmenters]
    if type(regions) == str:
        regions = [regions]
    assert len(localisers) == len(segmenters)
    num_models = len(localisers)
    # Broadcast parameters to length of localiser/segmenter model list.
    if type(loc_sizes) == tuple:
        loc_sizes = [loc_sizes] * num_models
    else:
        assert len(loc_sizes) == num_models
    if type(loc_spacings) == tuple:
        loc_spacings = [loc_spacings] * num_models
    else:
        assert len(loc_spacings) == num_models
    if type(pred_regions) == tuple:
        pred_regions = [pred_regions] * num_models
    else:
        assert len(pred_regions) == num_models
    if type(seg_spacings) == tuple:
        seg_spacings = [seg_spacings] * num_models
    else:
        assert len(seg_spacings) == num_models
    
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    region_data = patient.region_data(regions=regions) if regions is not None else None
    spacing = patient.ct_spacing

    # Load predictions.
    loc_centres = []
    preds = []
    for i in range(num_models):
        localiser = localisers[i]
        segmenter = segmenters[i]
        loc_size = loc_sizes[i]
        loc_spacing = loc_spacings[i]
        pred_region = pred_regions[i]
        seg_spacing = seg_spacings[i]

        # Load/make localiser prediction.
        loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser, raise_error=False)
        if loc_centre is None:
            logging.info(f"No localiser prediction found for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}', predicting...")
            truncate = True if pred_region == 'SpinalCord' else False
            create_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, truncate=truncate)
            loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)

        # Load/make segmenter prediction.
        pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter, raise_error=False)
        if pred is None:
            logging.info(f"No segmenter prediction found for dataset '{dataset}', patient '{pat_id}', localiser '{localiser}', segmenter '{segmenter}', predicting...")
            create_patient_segmenter_prediction(dataset, pat_id, pred_region, localiser, segmenter, seg_spacing)           # Handle multiple spacings.
            pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter)

        loc_centres.append(loc_centre)
        preds.append(pred)

    # Add 'centre_of' region data.
    if centre_of is not None:
        if centre_of == 'pred':
            if len(localisers) != 1:
                raise ValueError(f"Must pass model index (e.g. 'centre_of=pred:0' when plotting multiple predictions.")
            if region_data is None:
                region_data = {}
                region_data[centre_of] = preds[0]
        else:
            # Search for 'pred:0' pattern. 
            match = re.search(PRED_PATTERN, centre_of)
            if match is not None:
                pred_i = int(match.group(1))
                if pred_i >= len(localisers):
                    raise ValueError(f"Model index '{centre_of}' too high. Must pass index less than or equal to 'centre_of=pred:{len(localisers) - 1}'.")
                if region_data is None:
                    region_data = {}
                    region_data[centre_of] = preds[pred_i]
            else:
                # 'centre_of' is a region name.
                if region_data is None:
                    region_data = {}
                    centre_of_data = patient.region_data(regions=centre_of)
                    region_data[centre_of] = centre_of_data
                elif centre_of not in region_data.keys():
                    centre_of_data = patient.region_data(regions=centre_of)
                    region_data[centre_of] = centre_of_data[centre_of]

    # Add 'crop' region data.
    if type(crop) == str:
        if crop == 'pred':
            if len(localisers) != 1:
                raise ValueError(f"Must pass model index when plotting multiple predictions, e.g. 'crop=pred:0'.")
            if region_data is None:
                region_data = {}
                region_data[crop] = preds[0]
        else:
            # Search for 'pred:0' pattern. 
            match = re.search(PRED_PATTERN, crop)
            if match is not None:
                pred_i = int(match.group(1))
                if pred_i >= len(localisers):
                    raise ValueError(f"Model index '{crop}' too high. Must pass index less than or equal to 'crop=pred:{len(localisers) - 1}'.")
                if region_data is None:
                    region_data = {}
                    region_data[crop] = preds[pred_i]
            else:
                # 'crop' is a region name.
                if region_data is None:
                    region_data = {}
                    crop_data = patient.region_data(regions=crop)
                    region_data[crop] = crop_data
                elif crop not in region_data.keys():
                    crop_data = patient.region_data(regions=crop)
                    region_data[crop] = crop_data

    # Relabel regions.
    if region_labels is not None:
        # Names.
        regions = [region_labels[r] if r in region_labels else r for r in regions]
        # Region data.
        for old_label, new_label in region_labels.items():
            region_data[new_label] = region_data.pop(old_label)
        # Args.
        if centre_of is not None:
            centre_of = region_labels[centre_of]
        if crop is not None:
            if type(crop) == str:
                crop = region_labels[crop]
    
    # Plot.
    plot_segmenter_prediction(pat_id, spacing, preds, centre_of=centre_of, crop=crop, ct_data=ct_data, loc_centre=loc_centres, regions=regions, region_data=region_data, **kwargs)

from typing import List, Optional, Union

from mymi import dataset as ds
from mymi.geometry import get_extent_centre
from mymi.prediction.dataset.nifti import get_patient_localiser_prediction, get_patient_segmenter_prediction, load_patient_localiser_centre, load_patient_localiser_prediction, load_patient_segmenter_prediction
from mymi import types

from ..plotter import plot_localiser_prediction, plot_regions, plot_segmenter_prediction

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    regions: types.PatientRegions = 'all',
    show_dose: bool = False,
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=regions)
    spacing = patient.ct_spacing
    dose_data = patient.dose_data if show_dose else None
    
    # Plot.
    plot_regions(pat_id, ct_data, region_data, spacing, dose_data=dose_data, regions=regions, **kwargs)

def plot_patient_localiser_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localiser: types.ModelName,
    loc_size: types.ImageSize3D = (128, 128, 150),
    loc_spacing: types.ImageSpacing3D = (4, 4, 4),
    load_prediction: bool = True,
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=region)[region]
    spacing = patient.ct_spacing

    # Load prediction.
    if load_prediction:
        pred = load_patient_localiser_prediction(dataset, pat_id, localiser)
    else:
        # Set truncation if 'SpinalCord'.
        truncate = True if region == 'SpinalCord' else False

        # Make prediction.
        pred = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, truncate=truncate)
    
    # Plot.
    plot_localiser_prediction(pat_id, region, ct_data, region_data, spacing, pred, **kwargs)

def plot_patient_segmenter_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    localisers: Union[types.ModelName, List[types.ModelName]],
    segmenters: Union[types.ModelName, List[types.ModelName]],
    loc_sizes: Optional[Union[types.ImageSize3D, List[types.ImageSize3D]]] = (128, 128, 150),
    loc_spacing: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (4, 4, 4),
    seg_spacing: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (1, 1, 2),
    load_loc_prediction: bool = True,
    load_seg_prediction: bool = True,
    **kwargs) -> None:
    # Convert args to list.
    if type(localisers) == tuple:
        localisers = [localisers]
    if type(segmenters) == tuple:
        segmenters = [segmenters]
    assert len(localisers) == len(segmenters)
    # Broadcast sizes/spacings to model list length.
    if type(loc_sizes) == tuple:
        loc_sizes = [loc_sizes] * len(localisers)
    elif len(loc_sizes) == 1 and len(localisers) > 1:
        loc_sizes = loc_sizes * len(localisers)
    if type(loc_spacings) == tuple:
        loc_spacings = [loc_spacings] * len(localisers)
    elif len(loc_spacings) == 1 and len(localisers) > 1:
        loc_spacings = loc_spacings * len(localisers)
    if type(seg_spacings) == tuple:
        seg_spacings = [seg_spacings] * len(localisers)
    elif len(seg_spacings) == 1 and len(localisers) > 1:
        seg_spacings = seg_spacings * len(localisers)
    
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=region)[region]
    spacing = patient.ct_spacing

    # Load predictions.
    loc_centres = []
    preds = []
    for localiser, segmenter, loc_size, loc_spacing, seg_spacing in zip(localisers, segmenters, loc_sizes, loc_spacings, seg_spacings):
        if load_seg_prediction:
            loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
            pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter)
        else:
            if load_loc_prediction:
                loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
            else:
                # Set truncation if 'SpinalCord'.
                truncate = True if region == 'SpinalCord' else False

                pred = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, truncate=truncate)     # Handle multiple spacings.
                loc_centre = get_extent_centre(pred)

            # Make prediction.
            pred = get_patient_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, seg_spacing)           # Handle multiple spacings.

        loc_centres.append(loc_centre)
        preds.append(pred)
    
    # Plot.
    plot_segmenter_prediction(pat_id, region, ct_data, region_data, spacing, preds, loc_centre=loc_centres, **kwargs)

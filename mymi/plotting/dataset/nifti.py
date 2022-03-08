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
    **kwargs) -> None:
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=regions)
    spacing = patient.ct_spacing
    
    # Plot.
    plot_regions(pat_id, ct_data, region_data, spacing, regions=regions, **kwargs)

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

def plot_patient_segmenter_predictions(
    dataset: str,
    pat_id: str,
    region: str,
    localisers: Union[types.ModelName, List[types.ModelName]],
    segmenters: Union[types.ModelName, List[types.ModelName]],
    loc_sizes: Optional[Union[types.ImageSize3D, List[types.ImageSize3D]]] = (128, 128, 150),
    loc_spacings: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (4, 4, 4),
    seg_spacings: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (1, 1, 2),
    load_loc_prediction: bool = True,
    load_seg_prediction: bool = True,
    **kwargs) -> None:
    if type(localisers) == types.ModelName:
        localisers = [localisers]
    if type(segmenters) == types.ModelName:
        segmenters = [segmenters]
    assert len(localisers) == len(segmenters)
    
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(regions=region)[region]
    spacing = patient.ct_spacing

    # Load predictions.
    loc_centres = []
    preds = []
    for localiser, segmenter in zip(localisers, segmenters):
        if load_seg_prediction:
            loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
            pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter)
        else:
            if load_loc_prediction:
                loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
            else:
                # Set truncation if 'SpinalCord'.
                truncate = True if region == 'SpinalCord' else False

                pred = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_sizes, loc_spacings, truncate=truncate)     # Handle multiple spacings.
                loc_centre = get_extent_centre(pred)

            # Make prediction.
            pred = get_patient_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, seg_spacings)           # Handle multiple spacings.

        loc_centres.append(loc_centre)
        preds.append(pred)
    
    # Plot.
    plot_segmenter_prediction(pat_id, region, ct_data, loc_centres, region_data, spacing, preds, **kwargs)

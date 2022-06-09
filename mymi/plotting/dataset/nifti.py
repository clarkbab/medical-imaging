from typing import List, Optional, Union

from mymi import dataset as ds
from mymi.geometry import get_extent_centre
from mymi.prediction.dataset.nifti import get_patient_localiser_prediction, get_patient_segmenter_prediction, load_patient_localiser_centre, load_patient_localiser_prediction, load_patient_segmenter_prediction
from mymi import types

from ..plotter import plot_localiser_prediction, plot_regions, plot_segmenter_prediction

def plot_patient_regions(
    dataset: str,
    pat_id: str,
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    regions: Optional[types.PatientRegions] = None,
    show_dose: bool = False,
    **kwargs) -> None:
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
    if type(crop) == str:
        if region_data is None:
            crop_data = patient.region_data(regions=crop)
            region_data = crop_data
        elif crop not in region_data.keys():
            crop_data = patient.region_data(regions=crop)
            region_data[crop] = crop_data[crop]

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
    localiser: Union[types.ModelName, List[types.ModelName]],
    segmenter: Union[types.ModelName, List[types.ModelName]],
    centre_of: Optional[str] = None,
    crop: Optional[Union[str, types.Crop2D]] = None,
    load_loc_prediction: bool = True,
    load_seg_prediction: bool = True,
    loc_size: Optional[Union[types.ImageSize3D, List[types.ImageSize3D]]] = (128, 128, 150),
    loc_spacing: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (4, 4, 4),
    regions: Optional[types.PatientRegions] = None,
    show_ct: bool = True,
    seg_spacing: Optional[Union[types.ImageSpacing3D, List[types.ImageSpacing3D]]] = (1, 1, 2),
    **kwargs) -> None:
    # Convert args to list.
    if type(localiser) == tuple:
        localiser = [localiser]
    if type(segmenter) == tuple:
        segmenter = [segmenter]
    assert len(localiser) == len(segmenter)
    # Broadcast sizes/spacings to model list length.
    if type(loc_size) == tuple:
        loc_size = [loc_size] * len(localiser)
    elif len(loc_size) == 1 and len(localiser) > 1:
        loc_size = loc_size * len(localiser)
    if type(loc_spacing) == tuple:
        loc_spacing = [loc_spacing] * len(localiser)
    elif len(loc_spacing) == 1 and len(localiser) > 1:
        loc_spacing = loc_spacing * len(localiser)
    if type(seg_spacing) == tuple:
        seg_spacing = [seg_spacing] * len(localiser)
    elif len(seg_spacing) == 1 and len(localiser) > 1:
        seg_spacing = seg_spacing * len(localiser)
    
    # Load data.
    patient = ds.get(dataset, 'nifti').patient(pat_id)
    ct_data = patient.ct_data if show_ct else None
    region_data = patient.region_data(regions=regions) if regions is not None else None
    spacing = patient.ct_spacing

    # Load predictions.
    loc_centres = []
    preds = []
    for localiser, segmenter, loc_size, loc_spacing, seg_spacing in zip(localiser, segmenter, loc_size, loc_spacing, seg_spacing):
        if load_seg_prediction:
            loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
            pred = load_patient_segmenter_prediction(dataset, pat_id, localiser, segmenter)
        else:
            if load_loc_prediction:
                loc_centre = load_patient_localiser_centre(dataset, pat_id, localiser)
            else:
                # Set truncation if 'SpinalCord'.
                truncate = True if 'SpinalCord' in localiser[0] else False

                pred = get_patient_localiser_prediction(dataset, pat_id, localiser, loc_size, loc_spacing, truncate=truncate)     # Handle multiple spacings.
                loc_centre = get_extent_centre(pred)

            # Make prediction.
            pred = get_patient_segmenter_prediction(dataset, pat_id, region, loc_centre, segmenter, seg_spacing)           # Handle multiple spacings.

        loc_centres.append(loc_centre)
        preds.append(pred)

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
    plot_segmenter_prediction(pat_id, spacing, preds, centre_of=centre_of, crop=crop, ct_data=ct_data, loc_centre=loc_centres, regions=regions, region_data=region_data, **kwargs)

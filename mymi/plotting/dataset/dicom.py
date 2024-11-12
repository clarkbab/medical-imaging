from typing import Dict, List, Optional, Union

from ..plotter import plot_patient as plot_patient_base
from ..plotter import plot_segmenter_prediction
from mymi.dataset.dicom import DicomDataset
from mymi.prediction.dataset.dicom import load_segmenter_predictions
from mymi import types
from mymi.utils import arg_to_list

def plot_patient(
    dataset: str,
    pat_id: str,
    centre: Optional[str] = None,
    crop: Optional[Union[str, types.Box2D]] = None,
    regions: Optional[types.PatientRegions] = None,
    region_labels: Dict[str, str] = {},
    show_dose: bool = False,
    study_id: Optional[str] = None,
    use_mapping: bool = True,
    **kwargs) -> None:

    # Deal with 'regions' arg.
    patient = DicomDataset(dataset).patient(pat_id)
    if regions == 'all':
        regions = patient.list_regions()
    else:
        regions = arg_to_list(regions, str)

    if study_id is not None:
        study = patient.study(study_id)
    else:
        study = patient.default_study
    ct_data = study.ct_data
    region_data = study.region_data(regions=regions, use_mapping=use_mapping) if regions is not None else None
    spacing = study.ct_spacing
    dose_data = study.dose_data if show_dose else None

    if centre is not None:
        if type(centre) == str:
            if region_data is None or centre not in region_data:
                centre = study.region_data(regions=centre, use_mapping=use_mapping)[centre]

    if crop is not None:
        if type(crop) == str:
            if region_data is None or crop not in region_data:
                crop = study.region_data(regions=crop, use_mapping=use_mapping)[crop]

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
    plot_patient_base(pat_id, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, dose_data=dose_data, region_data=region_data, **kwargs)

def plot_model_prediction(
    dataset: str,
    pat_id: str,
    region: str,
    models: Union[str, List[str]],
    show_dose: bool = False,
    **kwargs) -> None:
    if type(models) == str:
        models = [models]
    
    # Load data.
    patient = DicomDataset(dataset).patient(pat_id)
    ct_data = patient.ct_data
    region_data = patient.region_data(region=region)[region]
    spacing = patient.ct_spacing
    dose_data = patient.dose_data if show_dose else None

    # Load model predictions.
    preds = []
    for model in models:
        pred = load_segmenter_predictions(dataset, pat_id, model, region)
        preds.append(pred)

    # Plot.
    plot_segmenter_prediction(pat_id, region, ct_data, region_data, spacing, preds, dose_data=dose_data, pred_labels=models, **kwargs)

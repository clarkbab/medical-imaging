from typing import Dict, List, Optional, Union

from mymi.datasets.dicom import DicomDataset
from mymi.predictions.datasets.dicom import load_segmenter_predictions
from mymi.typing import Box2D, Landmarks, PatientRegions

from ..plotting import plot_patient as plot_patient_base
from ..plotting import plot_segmenter_prediction

def plot_patients(
    dataset: str,
    pat_id: str,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    landmarks: Optional[Landmarks] = None,
    regions: Optional[PatientRegions] = None,
    region_labels: Dict[str, str] = {},
    show_dose: bool = False,
    study_id: Optional[str] = None,
    use_mapping: bool = True,
    **kwargs) -> None:

    # Load CT data.
    pat = DicomDataset(dataset).patient(pat_id)
    if study_id is not None:
        study = pat.study(study_id)
    else:
        study = pat.default_study
    ct_data = study.ct_data
    spacing = study.ct_spacing
    dose_data = study.dose_data if show_dose else None

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

    # If 'centre' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
    if centre is not None:
        if isinstance(centre, str):
            if study.has_landmark(centre) and landmark_data is not None and centre not in landmark_data['landmark-id']:
                centre = study.landmark_data(landmarks=centre)
            elif study.has_regions(centre) and region_data is not None and centre not in region_data:
                centre = study.region_data(regions=centre)[centre]

    # Load 'crop' as np.array (region label) or pd.Series (landmark).
    if crop is not None:
        if isinstance(crop, str):
            if study.has_landmark(crop) and landmark_data is not None and crop not in landmark_data['landmark-id']:
                crop = study.landmark_data(landmarks=crop)
            elif study.has_regions(crop) and region_data is not None and crop not in region_data:
                crop = study.region_data(regions=crop)[crop]

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
    max_chars = 10
    study_id = study.id
    if len(study_id) > max_chars:
        study_id = study_id[:max_chars]
    id = f"{pat_id}:{study_id}"
    plot_patient_base(id, ct_data.shape, spacing, centre=centre, crop=crop, ct_data=ct_data, dose_data=dose_data, landmark_data=landmark_data, region_data=region_data, **kwargs)

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

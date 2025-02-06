from typing import *

from mymi.datasets.dicom import DicomDataset
from mymi.predictions.datasets.dicom import load_segmenter_predictions
from mymi.typing import *
from mymi.utils import *

from ..plotting import apply_region_labels, plot_histogram, plot_patients_matrix, plot_segmenter_predictions

def plot_dataset_histogram(
    dataset: str,
    n_pats: Optional[int] = None,
    pat_ids: Optional[PatientIDs] = None,
    **kwargs) -> None:
    set = DicomDataset(dataset)
    if n_pats is not None:
        assert pat_ids is None
        pat_ids = set.list_patients()
        pat_ids = pat_ids[:n_pats]
    ct_data = [set.patient(pat_id).ct_data for pat_id in pat_ids]
    ct_data = [c.flatten() for c in ct_data]
    ct_data = np.concatenate(ct_data)
    plot_histogram(ct_data, **kwargs)

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
    set = DicomDataset(dataset)
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
    plot_segmenter_predictions(pat_id, region, ct_data, region_data, spacing, preds, dose_data=dose_data, pred_labels=models, **kwargs)

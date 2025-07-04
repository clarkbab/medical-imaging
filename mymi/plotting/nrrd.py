from typing import *

from mymi.datasets import NrrdDataset
from mymi.gradcam.dataset.nrrd import load_multi_segmenter_heatmap
from mymi.typing import *
from mymi.utils import *

from ..plotting import plot_histogram
from ..plotting import apply_region_labels, plot_histogram
from ..plotting import plot_heatmap as plot_heatmap_base
from ..plotting import plot_patients_matrix

MODEL_SELECT_PATTERN = r'^model:([0-9]+)$'
MODEL_SELECT_PATTERN_MULTI = r'^model(:([0-9]+))?:([a-zA-Z_]+)$'

def plot_patients(
    dataset: str,
    pat_ids: PatientIDs,
    centre: Optional[str] = None,
    crop: Optional[Union[str, PixelBox]] = None,
    landmarks: Optional[Landmarks] = None,
    region_labels: Dict[str, str] = {},
    regions: Optional[Regions] = None,
    show_dose: bool = False,
    study_id: Optional[StudyID] = None,
    **kwargs) -> None:
    pat_ids = arg_to_list(pat_ids, PatientID)

    # Load CT data.
    set = NrrdDataset(dataset)
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
            region_data = study.regions_data(regions=regions, **kwargs)
        else:
            region_data = None

        # Load landmarks.
        if landmarks is not None:
            landmark_data = study.landmarks_data(landmarks=landmarks, use_patient_coords=False, **kwargs)
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
                    centre = study.landmarks_data(landmarks=centre)
                elif study.has_regions(centre) and region_data is not None and centre not in region_data:
                    centre = study.regions_data(regions=centre)[centre]

        # If 'crop' isn't in 'landmark_data' or 'region_data', pass it to base plotter as np.ndarray, or pd.DataFrame.
        if crop is not None:
            if isinstance(crop, str):
                if study.has_landmark(crop) and landmark_data is not None and crop not in landmark_data['landmark-id']:
                    crop = study.landmarks_data(landmarks=crop)
                elif study.has_regions(crop) and region_data is not None and crop not in region_data:
                    crop = study.regions_data(regions=crop)[crop]

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

def plot_heatmap(
    dataset: str,
    pat_id: str,
    model: ModelName,
    region: str,
    layer: int,
    centre: Optional[str] = None,
    crop: Optional[Union[str, PixelBox]] = None,
    **kwargs) -> None:
    # Load data.
    set = NrrdDataset(dataset)
    pat = set.patient(pat_id)
    ct_data = pat.ct_data
    region_data = pat.regions_data(region=region)
    spacing = pat.ct_spacing

    # Load heatmap.
    heatmap = load_multi_segmenter_heatmap(dataset, pat_id, model, region, layer)

    if centre is not None:
        centre = pat.regions_data(region=centre)[centre]

    if type(crop) == str:
        crop = pat.regions_data(region=crop)[crop]
    
    # Plot.
    plot_heatmap_base(heatmap, spacing, centre=centre, crop=crop, ct_data=ct_data, region_data=region_data, **kwargs)

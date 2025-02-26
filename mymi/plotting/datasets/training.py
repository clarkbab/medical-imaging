from typing import *

from mymi.datasets import TrainingDataset
from mymi.typing import *
from mymi.utils import *

from ..plotting import *

def plot_dataset_histogram(
    dataset: str,
    n_samples: Optional[int] = None,
    sample_ids: Optional[PatientIDs] = None,
    split_id: Optional[SplitID] = None,
    **kwargs) -> None:
    set = TrainingDataset(dataset)
    if split_id is None:
        split = set.split(set.list_splits()[0])
    else:
        split = set.split(split_id)
    if n_samples is not None:
        assert sample_ids is None
        sample_ids = split.list_samples()
        sample_ids = sample_ids[:n_samples]
    inputs = [split.sample(s).input for s in sample_ids]
    inputs = np.concatenate([i.flatten() for i in inputs])
    plot_histogram(inputs, **kwargs)

def plot_patients(
    dataset: str,
    splits: TrainingSplit,
    sample_ids: SampleIDs,
    centre: Optional[str] = None,
    crop: Optional[Union[str, Box2D]] = None,
    landmarks: Optional[Landmarks] = None,
    region_labels: Dict[str, str] = {},
    regions: Optional[PatientRegions] = None,
    show_dose: bool = False,
    study_id: Optional[StudyID] = None,
    **kwargs) -> None:
    sample_ids = arg_to_list(sample_ids, SampleID)
    splits = arg_to_list(splits, TrainingSplit)
    if len(splits) == 1:
        splits = splits * len(samples_ids)
    else:
        assert len(splits) == sample_ids

    # Load CT data.
    set = TrainingDataset(dataset)
    plot_ids = []
    ct_datas = []
    spacings = []
    region_datas = []
    landmark_datas = []
    dose_datas = []
    centres = []
    crops = []
    for s, sam in zip(splits, sample_ids):
        sample = set.split(s).sample(sam)
        plot_id = f"{s}:{sam}"
        plot_ids.append(plot_id)
        ct_data = sample.input
        ct_datas.append(ct_data)
        spacing = set.spacing
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

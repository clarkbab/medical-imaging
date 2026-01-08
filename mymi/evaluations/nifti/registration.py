import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi.metrics import dice, distances, tre
from mymi.predictions.nifti import load_registration
from mymi.regions import regions_to_list
from mymi.transforms import sample
from mymi.typing import *
from mymi.utils import *
    
def load_registration_evaluation(
    dataset: str,
    model: ModelIDs = 'all',
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    set = NiftiDataset(dataset)
    filepath = os.path.join(set.path, 'data', 'evaluations', 'registration')
    all_models = list(sorted(os.listdir(filepath)))
    models = arg_to_list(model, ModelID, literals={ 'all': all_models })

    # Load regions evaluations.
    dfs = []
    for m in models:
        filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', m, 'regions.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            dfs.append(df)
    if len(dfs) > 0:
        regions_df = pd.concat(dfs, ignore_index=True)
    else:
        regions_df = None

    # Load landmarks evaluation.
    dfs = []
    for m in models:
        filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', m, 'landmarks.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            dfs.append(df)
    if len(dfs) > 0:
        landmarks_df = pd.concat(dfs, ignore_index=True)
    else:
        landmarks_df = None

    return landmarks_df, regions_df

def get_registration_landmark_evaluation(
    dataset: str,
    fixed_pat: PatientID,
    fixed_study: StudyID,
    fixed_series: SeriesID,
    moving_pat: PatientID,
    moving_study: StudyID,
    moving_series: SeriesID,
    landmark: LandmarkIDs,
    model: str) -> Tuple[Optional[List[str]], Optional[List[float]], Optional[List[float]]]:
    set = NiftiDataset(dataset)
    fixed_pat = set.patient(fixed_pat)
    fixed_study = fixed_pat.study(fixed_study)
    fixed_series = fixed_study.landmarks_series(fixed_series)
    moving_pat = set.patient(moving_pat)
    moving_study = moving_pat.study(moving_study)
    moving_series = moving_study.landmarks_series(moving_series)

    # Load moved (predicted) region data.
    _, _, moved_dose, moved_landmarks, _ = load_registration(dataset, fixed_pat.id, model, study=fixed_study.id, series=fixed_series.id, landmark=landmark, moving_pat=moving_pat.id, moving_study=moving_study.id, moving_series=moving_series.id)

    if moved_landmarks is not None:
        # Load moving landmarks.
        moving_landmarks = moving_study.landmarks_data(landmark=landmark, sample_dose=moving_study.has_dose)

        # Merge on landmark IDs.
        merged_df = moving_landmarks.merge(moved_landmarks, on=['landmark-id'], how='inner', suffixes=['_moving', '_moved'])
        moving_cols = [f'{c}_moving' for c in range(3)]
        moving_data = merged_df[moving_cols].to_numpy()
        moved_cols = [f'{c}_moved' for c in range(3)]
        moved_data = merged_df[moved_cols].to_numpy()

        # Calculate TRE.
        tres = tre(moving_data, moved_data)

        # Store landmark IDs.
        landmarks = merged_df['landmark-id'].tolist()
    else:
        tres = None

    # Calculate dose error - between moving dose at landmarks and moved dose at landmarks.
    if moving_study.has_dose is not None and moved_landmarks is not None:
        # # Sample moved dose at fixed landmarks.
        # fixed_landmarks = fixed_study.landmarks_data(landmarks=landmarks)
        # fixed_landmarks = sample(moved_dose, fixed_landmarks, spacing=fixed_study.ct_spacing, origin=fixed_study.ct_origin)
        # dose_errors = np.abs(fixed_landmarks['sample'] - moving_landmarks['dose']).tolist()

        # Rather than moving the dose (resampling) and then sampling at fixed landmarks (2x samplings),
        # we can just move the fixed landmarks and sample the moving dose (1x sampling).
        moving_landmarks = moving_study.landmarks_data(landmark=landmark, sample_dose=True)
        moved_landmarks = sample(moving_study.dose_data, moved_landmarks, spacing=moving_study.dose_spacing, origin=moving_study.dose_origin, landmarks_col='dose')
        assert np.all(moving_landmarks['landmark-id'].values == moved_landmarks['landmark-id'].values)
        dose_errors = list((moved_landmarks['dose'] - moving_landmarks['dose']).values)

        # Store landmark IDs.
        landmarks = moving_landmarks['landmark-id'].tolist()
    else:
        dose_errors = None

    if moved_landmarks is None and moved_dose is None:
        landmarks = None

    return landmarks, tres, dose_errors

def get_registration_region_evaluation(
    dataset: DatasetID,
    fixed_pat: PatientID,
    fixed_study: StudyID,
    fixed_series: SeriesID,
    moving_pat: PatientID,
    moving_study: StudyID,
    moving_series: SeriesID,
    region: RegionID,
    model: ModelID,
    ) -> List[Dict[str, float]]:
    set = NiftiDataset(dataset)
    fixed_pat = set.patient(fixed_pat)
    fixed_study = fixed_pat.study(fixed_study)
    fixed_series = fixed_study.regions_series(fixed_series)
    moving_pat = set.patient(moving_pat)
    moving_study = moving_pat.study(moving_study)
    moving_series = moving_study.regions_series(moving_series)

    # Load moved (predicted) region data.
    _, _, _, _, moved_regions_data = load_registration(dataset, fixed_pat.id, model, study=fixed_study.id, series=fixed_series.id, moving_pat=moving_pat.id, moving_study=moving_study.id, moving_series=moving_series.id, raise_error=False, region=region)
    if region not in moved_regions_data:
        return {}

    # Load fixed region data.
    fixed_regions_data = fixed_study.regions_series(fixed_series).data(region=region)

    # Get labels.
    pred = moved_regions_data[region]
    gt = fixed_regions_data[region]

    # # Only evaluate 'SpinalCord' up to the last common foreground slice in the caudal-z direction.
    # if region == 'SpinalCord':
    #     z_min_pred = np.nonzero(pred)[2].min()
    #     z_min_label = np.nonzero(label)[2].min()
    #     z_min = np.max([z_min_label, z_min_pred])

    #     # Crop pred/label foreground voxels.
    #     crop = ((0, 0, z_min), label.shape)
    #     pred = crop_foreground(pred, crop, use_patient_coords=False)
    #     label = crop_foreground(label, crop, use_patient_coords=False)

    # Dice.
    metrics = {}
    metrics['dice'] = dice(pred, gt)

    # Distances.
    if pred.sum() == 0 or gt.sum() == 0:
        metrics['apl'] = np.nan
        metrics['hd'] = np.nan
        metrics['hd-95'] = np.nan
        metrics['msd'] = np.nan
        metrics['surface-dice'] = np.nan
    else:
        # # Get distance metrics.
        # dists = distances(pred, gt, fixed_spacing)
        # for metric, value in dists.items():
        #     metrics[metric] = value

        # Add 'deepmind' comparison.
        dists = distances(pred, gt, fixed_study.ct_spacing)
        for m, v in dists.items():
            metrics[m] = v

    return metrics
    
def create_registration_evaluation(
    dataset: str,
    models: ModelIDs,
    exclude_pat: Optional[PatientIDs] = None,
    fixed_landmarks_series: SeriesID = 'series_1',
    fixed_regions_series: SeriesID = 'series_1',
    fixed_study: str = 'study_1',
    landmark: Optional[LandmarkIDs] = 'all',
    moving_study: str = 'study_0',
    moving_landmarks_series: SeriesID = 'series_1',
    moving_regions_series: SeriesID = 'series_1',
    pat: PatientIDs = 'all',
    region: Optional[RegionIDs] = 'all',
    group: Optional[PatientGroups] = 'all') -> None:
    models = arg_to_list(models, ModelID)

    # Add evaluations to dataframe.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(exclude=exclude_pat, group=group, pat=pat)
    fixed_study_id = fixed_study
    moving_study_id = moving_study

    # Evaluate registered regions.
    regions = regions_to_list(region, literals={ 'all': set.list_regions })
    for m in models:
        if regions is not None:
            logging.info(f"Evaluating region registrations for dataset '{dataset}' with model '{m}'...")

            cols = {
                'dataset': str,
                'patient-id': str,
                'study-id': str,
                'series-id': str,
                'moving-patient-id': str,
                'moving-study-id': str,
                'region-id': str,
                'model': str,
                'metric': str,
                'value': float
            }
            df = pd.DataFrame(columns=cols.keys())

            for p in tqdm(pat_ids):
                for r in tqdm(regions, leave=False):
                    # Skip if either moving/fixed study is missing the region.
                    moving_study = set.patient(p).study(moving_study_id)
                    moving_series = moving_study.regions_series(moving_series_series)
                    fixed_study = set.patient(p).study(fixed_study_id)
                    fixed_series = fixed_study.regions_series(fixed_regions_series)
                    if not fixed_series.has_region(r) or moving_series.has_region(r):
                        continue

                    # Get metrics per region.
                    metrics = get_registration_region_evaluation(dataset, p, fixed_study.id, fixed_series.id, p, moving_study.id, moving_series.id, r, m)
                    for met, v in metrics.items():
                        data = {
                            'dataset': dataset,
                            'patient-id': p,
                            'study-id': fixed_study.id,
                            'series-id': fixed_series.id,
                            'moving-patient-id': p,
                            'moving-study-id': moving_study.id,
                            'moving-series-id': moving_series.id,
                            'region-id': r,
                            'model': m,
                            'metric': met,
                            'value': v
                        }
                        df = append_row(df, data)

            if len(df) > 0:
                # Save evaluation.
                df = df.astype(cols)
                filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', m, 'regions.csv')
                save_csv(df, filepath, overwrite=True)

        if landmark is not None:
            logging.info(f"Evaluating landmark registrations for dataset '{dataset}' with model '{m}'...")

            cols = {
                'dataset': str,
                'patient-id': str,
                'study-id': str,
                'series-id': str,
                'moving-patient-id': str,
                'moving-study-id': str,
                'moving-series-id': str,
                'landmark-id': str,
                'model': str,
                'metric': str,
                'value': float
            }

            # Create dataframe.
            # Must have a single dataframe for all landmarks, as some patients have 300 landmarks!
            df = pd.DataFrame(columns=cols.keys())

            for p in tqdm(pat_ids):
                # Skip if either moving/fixed study is missing the landmarks.
                pat = set.patient(p)
                moving_study = pat.study(moving_study_id)
                moving_series = moving_study.landmarks_series(moving_landmarks_series)
                fixed_study = pat.study(fixed_study_id)
                fixed_series = fixed_study.landmarks_series(fixed_landmarks_series)
                if not fixed_series.has_landmark(landmark, any=True) or not moving_series.has_landmark(landmark, any=True):
                    continue

                # Get metrics per region.
                lm_ids, tres, dose_errors = get_registration_landmark_evaluation(dataset, p, fixed_study.id, fixed_series.id, p, moving_study.id, moving_series.id, landmark, m)
                if tres is not None:
                    for l, t in zip(lm_ids, tres):
                        data = {
                            'dataset': dataset,
                            'patient-id': p,
                            'study-id': fixed_study.id,
                            'series-id': fixed_series.id,
                            'moving-patient-id': p,
                            'moving-study-id': moving_study.id,
                            'moving-series-id': moving_series.id,
                            'landmark-id': l,
                            'model': m,
                            'metric': 'tre',
                            'value': t
                        }
                        df = append_row(df, data)

                if dose_errors is not None:
                    for l, e in zip(lm_ids, dose_errors):
                        data = {
                            'dataset': dataset,
                            'patient-id': p,
                            'study-id': fixed_study.id,
                            'series-id': fixed_series.id,
                            'moving-patient-id': p,
                            'moving-study-id': moving_study.id,
                            'moving-series-id': moving_series.id,
                            'landmark-id': l,
                            'model': m,
                            'metric': 'dose-error',
                            'value': e
                        }
                        df = append_row(df, data)

            if len(df) > 0:
                # Save evaluation.
                df = df.astype(cols)
                filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', m, 'landmarks.csv')
                save_csv(df, filepath, overwrite=True)

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
    
def load_registrations_evaluations(
    dataset: str,
    models: Union[str, List[str]]) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    set = NiftiDataset(dataset)
    models = arg_to_list(models, str)

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

    return regions_df, landmarks_df

def get_registration_landmarks_evaluation(
    dataset: str,
    fixed_pat_id: PatientID,
    fixed_study_id: str,
    moving_pat_id: PatientID,
    moving_study_id: str,
    landmark_ids: LandmarkIDs,
    model: str) -> Tuple[Optional[List[str]], Optional[List[float]], Optional[List[float]]]:
    set = NiftiDataset(dataset)
    fixed_pat = set.patient(fixed_pat_id)
    fixed_study = fixed_pat.study(fixed_study_id)
    moving_pat = set.patient(moving_pat_id)
    moving_study = moving_pat.study(moving_study_id)

    # Load moved (predicted) region data.
    _, _, _, moved_landmarks, moved_dose = load_registration(dataset, fixed_pat.id, model, fixed_study_id=fixed_study.id, landmark_ids=landmark_ids, moving_pat_id=moving_pat.id, moving_study_id=moving_study.id)

    if moved_landmarks is not None:
        # Load moving landmarks.
        moving_landmarks = moving_study.landmark_data(landmark_ids=landmark_ids, sample_dose=moving_study.has_dose)

        # Merge on landmark IDs.
        merged_df = moving_landmarks.merge(moved_landmarks, on=['landmark-id'], how='inner', suffixes=['_moving', '_moved'])
        moving_cols = [f'{c}_moving' for c in range(3)]
        moving_data = merged_df[moving_cols].to_numpy()
        moved_cols = [f'{c}_moved' for c in range(3)]
        moved_data = merged_df[moved_cols].to_numpy()

        # Calculate TRE.
        tres = tre(moving_data, moved_data)

        # Store landmark IDs.
        landmark_ids = merged_df['landmark-id'].tolist()
    else:
        tres = None

    # Calculate dose error - between moving dose at landmarks and moved dose at landmarks.
    if moving_study.has_dose is not None and moved_landmarks is not None:
        # # Sample moved dose at fixed landmarks.
        # fixed_landmarks = fixed_study.landmark_data(landmark_ids=landmark_ids)
        # fixed_landmarks = sample(moved_dose, fixed_landmarks, spacing=fixed_study.ct_spacing, offset=fixed_study.ct_offset)
        # dose_errors = np.abs(fixed_landmarks['sample'] - moving_landmarks['dose']).tolist()

        # Rather than moving the dose (resampling) and then sampling at fixed landmarks (2x samplings),
        # we can just move the fixed landmarks and sample the moving dose (1x sampling).
        moving_landmarks = moving_study.landmark_data(landmark_ids=landmark_ids, sample_dose=True)
        moved_landmarks = sample(moving_study.dose_data, moved_landmarks, spacing=moving_study.dose_spacing, offset=moving_study.dose_offset, landmarks_col='dose')
        assert np.all(moving_landmarks['landmark-id'].values == moved_landmarks['landmark-id'].values)
        dose_errors = list((moved_landmarks['dose'] - moving_landmarks['dose']).values)

        # Store landmark IDs.
        landmark_ids = moving_landmarks['landmark-id'].tolist()
    else:
        dose_errors = None

    if moved_landmarks is None and moved_dose is None:
        landmark_ids = None

    return landmark_ids, tres, dose_errors

def get_registration_region_evaluation(
    dataset: str,
    fixed_pat_id: PatientID,
    fixed_study_id: str,
    moving_pat_id: PatientID,
    moving_study_id: str,
    region_id: RegionID,
    model: str) -> List[Dict[str, float]]:
    set = NiftiDataset(dataset)
    fixed_pat = set.patient(fixed_pat_id)
    fixed_study = fixed_pat.study(fixed_study_id)
    moving_pat = set.patient(moving_pat_id)
    moving_study = moving_pat.study(moving_study_id)

    # Load moved (predicted) region data.
    _, _, moved_region_data, _, _ = load_registration(dataset, fixed_pat.id, model, fixed_study_id=fixed_study.id, moving_pat_id=moving_pat.id, moving_study_id=moving_study_id, raise_error=False, region_ids=region_id)
    if region_id not in moved_region_data:
        return {}

    # Load fixed region data.
    fixed_region_data = fixed_study.region_data(region_ids=region_id)

    # Get labels.
    pred = moved_region_data[region_id]
    gt = fixed_region_data[region_id]

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
    
def create_registration_evaluations(
    dataset: str,
    models: ModelIDs,
    exclude_pat_ids: Optional[PatientIDs] = None,
    fixed_study_id: str = 'study_1',
    landmark_ids: Optional[LandmarkIDs] = 'all',
    moving_study_id: str = 'study_0',
    pat_ids: PatientIDs = 'all',
    region_ids: Optional[RegionIDs] = 'all',
    splits: Optional[Splits] = 'all') -> None:
    models = arg_to_list(models, ModelID)

    # Add evaluations to dataframe.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(exclude=exclude_pat_ids, pat_ids=pat_ids, splits=splits)

    # Evaluate registered regions.
    region_ids = regions_to_list(region_ids, literals={ 'all': set.list_regions })
    for m in models:
        if region_ids is not None:
            logging.info(f"Evaluating region registrations for dataset '{dataset}' with model '{m}'...")

            cols = {
                'dataset': str,
                'fixed-patient-id': str,
                'fixed-study-id': str,
                'moving-patient-id': str,
                'moving-study-id': str,
                'region-id': str,
                'model': str,
                'metric': str,
                'value': float
            }
            df = pd.DataFrame(columns=cols.keys())

            for p in tqdm(pat_ids):
                for r in tqdm(region_ids, leave=False):
                    # Skip if either moving/fixed study is missing the region.
                    moving_study = set.patient(p).study(moving_study_id)
                    fixed_study = set.patient(p).study(fixed_study_id)
                    if not moving_study.has_regions(r) or not fixed_study.has_regions(r):
                        continue

                    # Get metrics per region.
                    metrics = get_registration_region_evaluation(dataset, p, fixed_study_id, p, moving_study_id, r, m)
                    for met, v in metrics.items():
                        data = {
                            'dataset': dataset,
                            'fixed-patient-id': p,
                            'fixed-study-id': fixed_study_id,
                            'moving-patient-id': p,
                            'moving-study-id': moving_study_id,
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

        if landmark_ids is not None:
            logging.info(f"Evaluating landmark registrations for dataset '{dataset}' with model '{m}'...")

            cols = {
                'dataset': str,
                'fixed-patient-id': str,
                'fixed-study-id': str,
                'landmark-id': str,
                'moving-patient-id': str,
                'moving-study-id': str,
                'model': str,
                'metric': str,
                'value': float
            }

            # Create dataframe.
            # Must have a single dataframe for all landmarks, as some patients have 300 landmarks!
            df = pd.DataFrame(columns=cols.keys())

            for p in tqdm(pat_ids):
                # Skip if either moving/fixed study is missing the landmarks.
                moving_study = set.patient(p).study(moving_study_id)
                fixed_study = set.patient(p).study(fixed_study_id)
                if not moving_study.has_landmarks(landmark_ids, any=True) or not fixed_study.has_landmarks(landmark_ids, any=True):
                    continue

                # Get metrics per region.
                lm_ids, tres, dose_errors = get_registration_landmarks_evaluation(dataset, p, fixed_study_id, p, moving_study_id, landmark_ids, m)
                if tres is not None:
                    for l, t in zip(lm_ids, tres):
                        data = {
                            'dataset': dataset,
                            'fixed-patient-id': p,
                            'fixed-study-id': fixed_study_id,
                            'landmark-id': l,
                            'moving-patient-id': p,
                            'moving-study-id': moving_study_id,
                            'model': m,
                            'metric': 'tre',
                            'value': t
                        }
                        df = append_row(df, data)

                if dose_errors is not None:
                    for l, e in zip(lm_ids, dose_errors):
                        data = {
                            'dataset': dataset,
                            'fixed-patient-id': p,
                            'fixed-study-id': fixed_study_id,
                            'landmark-id': l,
                            'moving-patient-id': p,
                            'moving-study-id': moving_study_id,
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

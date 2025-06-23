import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import *

from mymi.datasets import NiftiDataset
from mymi.metrics import dice, distances, tre
from mymi.predictions.datasets.nifti import load_registration
from mymi.regions import regions_to_list
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
    landmarks: Landmarks,
    model: str) -> Tuple[List[str], List[float]]:

    # Load moved (predicted) region data.
    res = load_registration(dataset, fixed_pat_id, model, fixed_study_id=fixed_study_id, landmarks=landmarks, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, raise_error=False)
    if res is None:
        return {}
    _, _, _, moved_landmarks = res

    # Load moving landmarks.
    set = NiftiDataset(dataset)
    moving_study = set.patient(moving_pat_id).study(moving_study_id)
    moving_landmarks = moving_study.landmark_data(landmarks=landmarks)

    # Get shared landmarks.
    # merged_df = moving_landmarks.merge(moved_landmarks, on=['patient-id', 'landmark-id'], how='inner', suffixes=['_moving', '_moved'])
    merged_df = moving_landmarks.merge(moved_landmarks, on=['landmark-id'], how='inner', suffixes=['_moving', '_moved'])
    moving_cols = [f'{c}_moving' for c in range(3)]
    moving_data = merged_df[moving_cols].to_numpy()
    moved_cols = [f'{c}_moved' for c in range(3)]
    moved_data = merged_df[moved_cols].to_numpy()

    landmark_ids = merged_df['landmark-id'].tolist()
    tres = tre(moving_data, moved_data)

    return landmark_ids, tres

def get_registration_region_evaluation(
    dataset: str,
    fixed_pat_id: PatientID,
    fixed_study_id: str,
    moving_pat_id: PatientID,
    moving_study_id: str,
    region: Region,
    model: str) -> List[Dict[str, float]]:

    # Load moved (predicted) region data.
    res = load_registration(dataset, fixed_pat_id, model, fixed_study_id=fixed_study_id, moving_pat_id=moving_pat_id, moving_study_id=moving_study_id, raise_error=False, regions=region)
    if res is None:
        return {}
    _, _, moved_region_data, _ = res

    # Load fixed region data.
    fixed_study = NiftiDataset(dataset).patient(fixed_pat_id).study(fixed_study_id)
    fixed_spacing = fixed_study.ct_spacing
    fixed_region_data = fixed_study.region_data(regions=region)

    # Get labels.
    pred = moved_region_data[region]
    gt = fixed_region_data[region]

    # # Only evaluate 'SpinalCord' up to the last common foreground slice in the caudal-z direction.
    # if region == 'SpinalCord':
    #     z_min_pred = np.nonzero(pred)[2].min()
    #     z_min_label = np.nonzero(label)[2].min()
    #     z_min = np.max([z_min_label, z_min_pred])

    #     # Crop pred/label foreground voxels.
    #     crop = ((0, 0, z_min), label.shape)
    #     pred = crop_foreground_vox(pred, crop)
    #     label = crop_foreground_vox(label, crop)

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
        dists = distances(pred, gt, fixed_spacing)
        for m, v in dists.items():
            metrics[m] = v

    return metrics
    
def create_registrations_evaluation(
    dataset: str,
    models: Union[str, List[str]],
    exclude_pat_ids: Optional[PatientIDs] = None,
    fixed_study_id: str = 'study_1',
    landmarks: Optional[Landmarks] = 'all',
    moving_study_id: str = 'study_0',
    pat_ids: PatientIDs = 'all',
    regions: Optional[Regions] = 'all',
    splits: Optional[Splits] = 'all') -> None:
    models = arg_to_list(models, str)

    # Add evaluations to dataframe.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients(exclude=exclude_pat_ids, pat_ids=pat_ids, splits=splits)

    for m in models:
        if regions is not None:
            regions = regions_to_list(regions, literals={ 'all': set.list_regions })

            cols = {
                'dataset': str,
                'fixed-patient-id': str,
                'fixed-study-id': str,
                'moving-patient-id': str,
                'moving-study-id': str,
                'region': str,
                'model': str,
                'metric': str,
                'value': float
            }
            df = pd.DataFrame(columns=cols.keys())

            for r in tqdm(regions):
                for p in tqdm(pat_ids, leave=False):
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
                            'region': r,
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

    if landmarks is not None:
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

        for p in tqdm(pat_ids, leave=False):
            # Skip if either moving/fixed study is missing the landmarks.
            moving_study = set.patient(p).study(moving_study_id)
            fixed_study = set.patient(p).study(fixed_study_id)
            if not moving_study.has_landmarks(landmarks) or not fixed_study.has_landmarks(landmarks):
                continue

            # Get metrics per region.
            lm_ids, tres = get_registration_landmarks_evaluation(dataset, p, fixed_study_id, p, moving_study_id, landmarks, m)
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

        if len(df) > 0:
            # Save evaluation.
            df = df.astype(cols)
            filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', m, 'landmarks.csv')
            save_csv(df, filepath, overwrite=True)

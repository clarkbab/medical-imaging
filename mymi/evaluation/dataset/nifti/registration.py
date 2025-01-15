import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from mymi.dataset import NiftiDataset
from mymi.metrics import dice, distances_deepmind, tre
from mymi import logging
from mymi.prediction.dataset.nifti import load_registration
from mymi.regions import regions_to_list
from mymi.types import PatientID, PatientLandmarks, PatientRegion, PatientRegions
from mymi.utils import append_row, arg_to_list, load_csv, save_csv
    
def load_registration_evaluation(
    dataset: str,
    model: str,
    landmarks: Optional[PatientLandmarks] = None,
    regions: Optional[PatientRegions] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    set = NiftiDataset(dataset)

    # Load regions evaluations.
    if regions is not None:
        dfs = []
        regions = regions_to_list(regions, literals={ 'all': set.list_regions })
        for r in regions:
            filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', model, 'regions', f'{r}.csv')
            if not os.path.exists(filepath):
                continue
            df = load_csv(filepath)
            dfs.append(df)
        regions_eval = pd.concat(dfs, axis=0)
    else:
        regions_eval = None

    # Load landmarks evaluation.
    if landmarks is not None:
        filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', model, 'landmarks.csv')
        landmarks_eval = pd.read_csv(filepath)
    else:
        landmarks_eval = None

    return regions_eval, landmarks_eval

def get_registration_landmarks_evaluation(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: str,
    fixed_pat_id: PatientID,
    fixed_study_id: str,
    landmarks: PatientLandmarks,
    model: str) -> List[Dict[str, float]]:

    # Load moved (predicted) region data.
    _, _, _, moved_landmarks = load_registration(dataset, moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, landmarks=landmarks)

    # Load moving landmarks.
    set = NiftiDataset(dataset)
    moving_study = set.patient(moving_pat_id).study(moving_study_id)
    moving_spacing = moving_study.ct_spacing
    moving_landmarks = moving_study.landmark_data(landmarks=landmarks)

    # Get shared landmarks.
    merged_df = moving_landmarks.merge(moved_landmarks, on=['patient-id', 'study-id', 'landmark-id'], how='inner', suffixes=['_moving', '_moved'])
    moving_cols = [f'{c}_moving' for c in range(3)]
    moving_data = merged_df[moving_cols].to_numpy()
    moved_cols = [f'{c}_moved' for c in range(3)]
    moved_data = merged_df[moved_cols].to_numpy()

    tre_metrics = tre(moving_data, moved_data, moving_spacing)

    return tre_metrics

def get_registration_region_evaluation(
    dataset: str,
    moving_pat_id: PatientID,
    moving_study_id: str,
    fixed_pat_id: PatientID,
    fixed_study_id: str,
    region: PatientRegion,
    model: str) -> List[Dict[str, float]]:

    # Load moved (predicted) region data.
    _, _, moved_region_data, _ = load_registration(dataset, moving_pat_id, moving_study_id, fixed_pat_id, fixed_study_id, model, regions=region)

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
    #     pred = crop_foreground_3D(pred, crop)
    #     label = crop_foreground_3D(label, crop)

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
        # dists = all_distances(pred, gt, fixed_spacing)
        # for metric, value in dists.items():
        #     metrics[metric] = value

        # Add 'deepmind' comparison.
        dists = distances_deepmind(pred, gt, fixed_spacing)
        for m, v in dists.items():
            metrics[m] = v

    return metrics
    
def create_registration_evaluation(
    dataset: str,
    moving_study_id: str,
    fixed_study_id: str,
    model: str,
    landmarks: Optional[PatientLandmarks] = None,
    regions: Optional[PatientRegions] = None) -> None:
    logging.arg_log('Evaluating NIFTI registration', ('dataset', 'region'), (dataset, regions))

    # Add evaluations to dataframe.
    set = NiftiDataset(dataset)
    pat_ids = set.list_patients()
    if regions is not None:
        regions = regions_to_list(regions, literals={ 'all': set.list_regions })

        cols = {
            'dataset': str,
            'moving-patient-id': str,
            'moving-study-id': str,
            'fixed-patient-id': str,
            'fixed-study-id': str,
            'region': str,
            'model': str,
            'metric': str,
            'value': float
        }

        for r in tqdm(regions):

            # Create dataframe.
            # Save a separate dataframe per region as this means we don't have to
            # hash region names to create the unique filepath. Also eliminates redundancy
            # if we performed two separate evaluations with overlapping region names.
            df = pd.DataFrame(columns=cols.keys())

            for p in tqdm(pat_ids, leave=False):
                # Skip if either moving/fixed study is missing the region.
                moving_study = set.patient(p).study(moving_study_id)
                fixed_study = set.patient(p).study(fixed_study_id)
                if not moving_study.has_regions(r) or not fixed_study.has_regions(r):
                    continue

                # Get metrics per region.
                metrics = get_registration_region_evaluation(dataset, p, moving_study_id, p, fixed_study_id, r, model)
                for m, v in metrics.items():
                    data = {
                        'dataset': dataset,
                        'moving-patient-id': p,
                        'moving-study-id': moving_study_id,
                        'fixed-patient-id': p,
                        'fixed-study-id': fixed_study_id,
                        'region': r,
                        'model': model,
                        'metric': m,
                        'value': v
                    }
                    df = append_row(df, data)

            if len(df) > 0:
                # Save evaluation.
                df = df.astype(cols)
                filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', model, 'regions', f'{r}.csv')
                save_csv(df, filepath, overwrite=True)

    if landmarks is not None:
        cols = {
            'dataset': str,
            'moving-patient-id': str,
            'moving-study-id': str,
            'fixed-patient-id': str,
            'fixed-study-id': str,
            'model': str,
            'metric': str,
            'value': float
        }

        # Create dataframe.
        # Must have a single dataframe for all landmarks, as some patients have 300 landmarks!
        df = pd.DataFrame(columns=cols.keys())

        for p in tqdm(pat_ids, leave=False):
            # Skip if either moving/fixed study is missing the landmark.
            moving_study = set.patient(p).study(moving_study_id)
            fixed_study = set.patient(p).study(fixed_study_id)
            if not moving_study.has_landmark(landmarks) or not fixed_study.has_landmark(landmarks):
                continue

            # Get metrics per region.
            metrics = get_registration_landmarks_evaluation(dataset, p, moving_study_id, p, fixed_study_id, landmarks, model)
            for m, v in metrics.items():
                data = {
                    'dataset': dataset,
                    'moving-patient-id': p,
                    'moving-study-id': moving_study_id,
                    'fixed-patient-id': p,
                    'fixed-study-id': fixed_study_id,
                    'model': model,
                    'metric': m,
                    'value': v
                }
                df = append_row(df, data)

        if len(df) > 0:
            # Save evaluation.
            df = df.astype(cols)
            filepath = os.path.join(set.path, 'data', 'evaluations', 'registration', model, 'landmarks.csv')
            save_csv(df, filepath, overwrite=True)

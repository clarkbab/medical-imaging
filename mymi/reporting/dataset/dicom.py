import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import List

from mymi.dataset.dicom import DICOMDataset
from mymi.evaluation.dataset.dicom import evaluate_model
from mymi.geometry import get_extent
from mymi import types
from mymi.utils import append_row, encode

def create_evaluation_report(
    name: str,
    dataset: str,
    localiser: types.Model,
    segmenter: types.Model,
    region: str) -> None:
    # Save report.
    eval_df = evaluate_model(dataset, localiser, segmenter, region)
    set = DICOMDataset(dataset)
    filename = f"{name}.csv"
    filepath = os.path.join(set.path, 'reports', 'evaluation', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    eval_df.to_csv(filepath)

def get_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # Get patients.
    set = DICOMDataset(dataset)
    pats = set.list_patients(regions=regions)

    cols = {
        'patient-id': str,
        'axis': int,
        'size': int,
        'spacing': float,
        'fov': float
    }
    df = pd.DataFrame(columns=cols.keys())

    for pat in tqdm(pats):
        # Load values.
        patient = set.patient(pat)
        size = patient.ct_size()
        spacing = patient.ct_spacing()

        # Calculate FOV.
        fov = np.array(size) * spacing

        for axis in range(len(size)):
            data = {
                'patient-id': pat,
                'axis': axis,
                'size': size[axis],
                'spacing': spacing[axis],
                'fov': fov[axis]
            }
            df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    # Get summary.
    df = get_ct_summary(dataset, regions=regions)

    # Save summary.
    set = DICOMDataset(dataset)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_ct_summary(
    dataset: str,
    regions: types.PatientRegions = 'all') -> None:
    set = DICOMDataset(dataset)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    return pd.read_csv(filepath)

def get_patient_regions(dataset: str) -> pd.DataFrame:
    # List patients.
    set = DICOMDataset(dataset)
    pat_ids = set.list_patients()

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Add rows.
    for pat_id in tqdm(pat_ids):
        pat_regions = set.patient(pat_id).list_regions()
        for pat_region in pat_regions:
            data = {
                'patient-id': pat_id,
                'region': pat_region
            }
            df = append_row(df, data)

    return df

def create_patient_regions_report(dataset: str) -> None:
    # Generate counts report.
    pr_df = get_patient_regions(dataset)
    set = DICOMDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pr_df.to_csv(filepath, index=False)

def load_patient_regions_report(dataset: str) -> None:
    set = DICOMDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    return pd.read_csv(filepath)

def region_overlap(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> int:
    # List regions.
    set = DICOMDataset(dataset)
    regions_df = set.list_regions(clear_cache=clear_cache) 
    regions_df = regions_df.drop_duplicates()
    regions_df['count'] = 1
    regions_df = regions_df.pivot(index='patient-id', columns='region', values='count')

    # Filter on requested regions.
    def filter_fn(row):
        if type(regions) == str:
            if regions == 'all':
                return True
            else:
                return row[regions] == 1
        else:
            keep = True
            for region in regions:
                if row[region] != 1:
                    keep = False
            return keep
    regions_df = regions_df[regions_df.apply(filter_fn, axis=1)]
    return len(regions_df) 

def region_summary(
    dataset: str,
    regions: List[str]) -> pd.DataFrame:
    """
    returns: stats on region shapes.
    """
    set = DICOMDataset(dataset)
    pats = set.list_patients(regions=regions)

    cols = {
        'patient': str,
        'region': str,
        'axis': str,
        'extent-mm': float,
        'spacing-mm': float
    }
    df = pd.DataFrame(columns=cols.keys())

    axes = [0, 1, 2]

    # Initialise empty data structure.
    data = {}
    for region in regions:
        data[region] = {}
        for axis in axes:
            data[region][axis] = []

    for pat in tqdm(pats):
        # Get spacing.
        spacing = set.patient(pat).ct_spacing()

        # Get region data.
        pat_regions = set.patient(pat).list_regions(whitelist=regions)
        rs_data = set.patient(pat).region_data(regions=pat_regions)

        # Add extents for all regions.
        for r in rs_data.keys():
            r_data = rs_data[r]
            min, max = get_extent(r_data)
            for axis in axes:
                extent_vox = max[axis] - min[axis]
                extent_mm = extent_vox * spacing[axis]
                data = {
                    'patient': pat,
                    'region': r,
                    'axis': axis,
                    'extent-mm': extent_mm,
                    'spacing-mm': spacing[axis]
                }
                df = append_row(df, data)

    # Set column types as 'append' crushes them.
    df = df.astype(cols)

    return df

def create_region_summary_report(
    dataset: str,
    regions: List[str]) -> None:
    # Generate counts report.
    df = region_summary(dataset, regions)

    # Save report.
    filename = 'region-summary.csv'
    set = DICOMDataset(dataset)
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

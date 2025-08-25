from collections import Counter
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
import pytorch_lightning as pl
from tqdm import tqdm
from typing import List, Optional, Union

from mymi.datasets.dicom import DicomDataset
from mymi.geometry import fov
from mymi import logging
from mymi.regions import regions_to_list
from mymi.typing import Regions
from mymi.utils import append_row, encode

def create_evaluation_report(
    name: str,
    dataset: str,
    localiser: pl.LightningModule,
    segmenter: pl.LightningModule,
    region: str) -> None:
    # Save report.
    eval_df = evaluate_model(dataset, localiser, segmenter, region)
    set = DicomDataset(dataset)
    filename = f"{name}.csv"
    filepath = os.path.join(set.path, 'reports', 'evaluation', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    eval_df.to_csv(filepath)

def get_ct_summary(
    dataset: str,
    regions: Regions = 'all') -> pd.DataFrame:
    # Get patients.
    set = DicomDataset(dataset)
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
    regions: Regions = 'all') -> None:
    # Get summary.
    df = get_ct_summary(dataset, regions=regions)

    # Save summary.
    set = DicomDataset(dataset)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_ct_summary(
    dataset: str,
    regions: Regions = 'all') -> None:
    set = DicomDataset(dataset)
    filepath = os.path.join(set.path, 'reports', f'ct-summary-{encode(regions)}.csv')
    return pd.read_csv(filepath)

def get_patient_regions_report(
    dataset: str,
    use_mapping: bool = True) -> pd.DataFrame:
    # List patients.
    set = DicomDataset(dataset)
    pat_ids = set.list_patients()

    # Create dataframe.
    cols = {
        'patient-id': str,
        'study-id': str,
        'series-id': str,
        'rtstruct-id': str,
        'region': str,
    }
    df = pd.DataFrame(columns=cols.keys())

    # Add rows.
    for p in tqdm(pat_ids):
        pat = set.patient(p)
        def_study = pat.default_study
        def_rtstruct = def_study.default_series('rtstruct')
        if def_rtstruct is None:
            continue
        pat_regions = def_rtstruct.list_regions(use_mapping=use_mapping)
        for r in pat_regions:
            data = {
                'patient-id': p,
                'study-id': def_study.id,
                'rtstruct-series-id': def_rtstruct.id,
                'region': r,
            }
            df = append_row(df, data)

    return df

def create_patient_regions_report(
    dataset: str,
    use_mapping: bool = True) -> None:
    logging.info(f"Creating patient regions report for dataset '{dataset}' with mapping={use_mapping}.")
    df = get_patient_regions_report(dataset, use_mapping=use_mapping)
    set = DicomDataset(dataset)
    filename = 'regions-count.csv' if use_mapping else 'unmapped-regions-count.csv'
    filepath = os.path.join(set.path, 'data', 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def load_patient_regions_report(
    dataset: str,
    exists_only: bool = False,
    use_mapping: bool = True) -> Union[DataFrame, bool]:
    set = DicomDataset(dataset)
    filename = 'regions-count.csv' if use_mapping else 'unmapped-regions-count.csv'
    filepath = os.path.join(set.path, 'data', 'reports', filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath) if not exists_only else True
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Patient regions report doesn't exist for dataset '{dataset}'.")

def get_mapped_duplicates(dataset: str) -> DataFrame:
    # Allows us to check 'region-map.csv' mapping for duplicates rather than running 
    # 'create_patient_regions_report(..., use_mapping=True)' which will break on each duplicate.
    region_map = DicomDataset(dataset).region_map
    df = load_patient_regions_report(dataset, use_mapping=False)
    df['mapped'] = df[['patient-id', 'region']].apply(lambda row: region_map.to_internal(row['region'], pat_id=row['patient-id'])[0], axis=1)
    df = df.groupby('patient-id')['mapped'].apply(list).reset_index()
    df['mapped'] = df['mapped'].apply(lambda regions: [i for i, count in Counter(regions).items() if count > 1])
    df['duplicates'] = df['mapped'].apply(lambda dups: len(dups) > 0)
    df = df[df['duplicates']]
    return df

def get_regions_like(
    dataset: str,
    text: str,
    case: bool = False,
    use_mapping: bool = False) -> DataFrame:
    df = load_patient_regions_report(dataset, use_mapping=use_mapping)
    count_df = df.groupby('region')['patient-id'].count().rename('count').reset_index()
    if case:
        count_df = count_df[count_df['region'].str.contains(text)]
    else:
        count_df = count_df[count_df['region'].str.lower().str.contains(text.lower())]
    count_df = count_df.sort_values('count', ascending=False)
    return count_df

def region_overlap(
    dataset: str,
    clear_cache: bool = True,
    regions: Regions = 'all') -> int:
    # List regions.
    set = DicomDataset(dataset)
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

def get_region_summary(
    dataset: str,
    region: Regions) -> pd.DataFrame:
    set = DicomDataset(dataset)
    pat_ids = set.list_patients(region=region)

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

    for pat_id in tqdm(pat_ids):
        # Get spacing.
        pat = set.patient(pat_id)
        spacing = pat.ct_spacing()

        # Get region data.
        pat_regions = set.patient(pat).list_regions(whitelist=regions)
        rs_data = set.patient(pat).region_data(region=pat_regions)

        # Add extents for all regions.
        for r in rs_data.keys():
            r_data = rs_data[r]
            min, max = extent(r_data)
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

def create_region_counts(dataset: str) -> None:
    count_df = get_region_counts(dataset)
    set = DicomDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    count_df.to_csv(filepath, index=False)

def create_region_summary(
    dataset: str,
    regions: List[str]) -> None:
    # Generate summary report.
    df = region_summary(dataset, regions)

    # Save report.
    filename = 'region-summary.csv'
    set = DicomDataset(dataset)
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

def get_region_counts(dataset: str) -> DataFrame:
    # List patients.
    set = DicomDataset(dataset)
    pat_ids = set.list_patients()

    # Create dataframe.
    cols = {
        'patient-id': str,
        'region': str
    }
    df = DataFrame(columns=cols.keys())

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

def load_region_counts(
    dataset: str,
    regions: Optional[Regions] = None,
    exists_only: bool = False) -> Union[DataFrame, bool]:
    set = DicomDataset(dataset)
    filepath = os.path.join(set.path, 'reports', 'region-count.csv')
    if os.path.exists(filepath):
        if exists_only:
            return True
        else:
            df = pd.read_csv(filepath)
            df = df.astype({ 'patient-id': str })
            if regions is not None:
                regions = regions_to_list(regions)
                df = df[df['region'].isin(regions)]
            return df
    else:
        if exists_only:
            return False
        else:
            raise ValueError(f"Patient regions report doesn't exist for dataset '{dataset}'.")

import os
import pandas as pd

from mymi import dataset as ds
from mymi.evaluation.dataset.raw.nifti import evaluate_model
from mymi import types

def create_evaluation_report(
    name: str,
    dataset: str,
    localiser: types.Model,
    segmenter: types.Model,
    region: str) -> None:
    # Save report.
    eval_df = evaluate_model(dataset, localiser, segmenter, region)
    set = ds.get(dataset, 'dicom')
    filename = f"{name}.csv"
    filepath = os.path.join(set.path, 'reports', 'evaluation', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    eval_df.to_csv(filepath)

def region_count(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # List regions.
    set = ds.get(dataset, type_str='nifti')
    regions_df = set.list_regions(clear_cache=clear_cache)

    # Filter on requested regions.
    def filter_fn(row):
        if type(regions) == str:
            if regions == 'all':
                return True
            else:
                return row['region'] == regions
        else:
            for region in regions:
                if row['region'] == region:
                    return True
            return False
    regions_df = regions_df[regions_df.apply(filter_fn, axis=1)]

    # Generate counts report.
    count_df = regions_df.groupby('region').count().rename(columns={'patient-id': 'count'})
    return count_df

def create_region_count_report(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> None:
    # Generate counts report.
    set = ds.get(dataset, type_str='nifti')
    count_df = region_count(dataset, clear_cache=clear_cache, regions=regions)
    filename = 'region-count.csv'
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    count_df.to_csv(filepath)

def region_overlap(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> int:
    # List regions.
    set = ds.get(dataset, type_str='nifti')
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

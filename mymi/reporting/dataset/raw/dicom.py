import os
import pandas as pd

from mymi import dataset as ds
from mymi import types

def region_count(
    dataset: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # List regions.
    set = ds.get(dataset, type_str='dicom')
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
    count_df = regions_df.groupby('region').count()['patient-id']
    return count_df

def create_region_count_report(
    dataset: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> None:

    # Generate counts report.
    count_df = region_count(dataset, clear_cache=clear_cache, regions=regions)
    filename = 'region-count.csv'
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    count_df.to_csv(filepath)

def region_overlap(
    dataset: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> int:
    set = ds.get(dataset, type_str='dicom')
    # List regions.
    regions_df = set.list_regions(clear_cache=clear_cache) 
    regions_df = regions_df.drop_duplicates()

    # Filter on requested regions.
    def filter_fn(row):
        if type(regions) == str:
            if regions == 'all':
                return True
            else:
                return row['region'] == regions
        else:
            keep = True
            for region in regions:
                if row['region'] != region:
                    keep = False
    regions_df = regions_df[regions_df.apply(filter_rn, axis=1)]

    regions_df['count'] = 1
    pivot_df = regions_df.pivot(index='patient-id', columns='region', values='count')
    return len(pat_df) 

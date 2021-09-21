import os
import pandas as pd

from mymi import dataset as ds
from mymi import types

def region_count(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
    # List regions.
    set = ds.get(dataset, type_str='processed')
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
    count_df = regions_df.groupby(['partition', 'region']).count().rename(columns={'sample-index': 'count'})

    # Add 'p' column.
    count_df = count_df.reset_index()
    total_df = count_df.groupby('region').sum().rename(columns={'count': 'total'})
    count_df = count_df.join(total_df, on='region')
    count_df['p'] = count_df['count'] / count_df['total']
    count_df = count_df.drop(columns='total')
    count_df = count_df.set_index(['partition', 'region'])
    return count_df

def create_region_count_report(
    dataset: str,
    clear_cache: bool = True,
    regions: types.PatientRegions = 'all') -> None:
    # Generate counts report.
    set = ds.get(dataset, type_str='processed')
    count_df = region_count(dataset, clear_cache=clear_cache, regions=regions)
    filename = 'region-count.csv'
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    count_df.to_csv(filepath)

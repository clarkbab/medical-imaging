import os

from mymi import dataset as ds
from mymi import types

def create_region_count_report(dataset: str) -> None:
    set = ds.get(dataset, type_str='dicom')
    regions_df = set.list_regions()
    counts = regions_df.groupby('region').count()['patient-id']
    filename = 'region-count.csv'
    filepath = os.path.join(set.path, 'reports', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    counts.to_csv(filepath)

def get_region_overlap(
    dataset: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> int:
    set = ds.get(dataset, type_str='dicom')
    regions_df = set.list_regions(clear_cache=clear_cache) 
    regions_df = regions_df.drop_duplicates()
    regions_df['count'] = 1
    pivot_df = regions_df.pivot(index='patient-id', columns='region', values='count')

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

    pat_df = pivot_df[pivot_df.apply(filter_fn, axis=1)]
    return len(pat_df) 

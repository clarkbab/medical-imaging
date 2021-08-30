import pandas as pd

from mymi import types

def region_count(
    dataset: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:

def partition_region_count(
    dataset: str,
    partition: str,
    clear_cache: bool = False,
    regions: types.PatientRegions = 'all') -> pd.DataFrame:
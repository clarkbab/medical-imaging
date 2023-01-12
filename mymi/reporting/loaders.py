import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Union

from mymi import config
from mymi import dataset as ds
from mymi.loaders import Loader, MultiLoader
from mymi import logging
from mymi.types import PatientID
from mymi.utils import append_row, arg_log, arg_to_list, encode, load_csv, save_csv

def get_loader_manifest(
    dataset: Union[str, List[str]],
    region: str,
    check_processed: bool = True,
    n_folds: Optional[int] = 5,
    n_train: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    datasets = arg_to_list(dataset, str)

    # Create empty dataframe.
    cols = {
        'region': str,
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': str,
        'origin-dataset': str,
        'origin-patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, ds.get(d, 'training', check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    tl, vl, tsl = Loader.build_loaders(datasets, region, check_processed=check_processed, load_data=False, load_test_origin=False, n_folds=n_folds, n_train=n_train, shuffle_train=False, test_fold=test_fold)
    loader_names = ['train', 'validate', 'test']

    # Get values for this region.
    for loader_name, loader in zip(loader_names, (tl, vl, tsl)):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                origin_ds, origin_pat_id = dataset_map[dataset].sample(sample_id).origin
                data = {
                    'region': region,
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def create_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    check_processed: bool = True,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    logging.info(f"Creating loader manifest for datasets '{datasets}', region '{region}', n_folds '{n_folds}', test_fold '{test_fold}'.")

    # Get manifest.
    df = get_loader_manifest(datasets, region, check_processed=check_processed, n_folds=n_folds, test_fold=test_fold)

    # Save manifest.
    save_csv(df, 'loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv', index=False, overwrite=True)

def load_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    test_fold: Optional[int] = None) -> pd.DataFrame:
    df = load_csv('loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv')
    df = df.astype({ 'origin-patient-id': str, 'sample-id': str })
    return df

def get_test_fold(
    datasets: Union[str, List[str]],
    dataset: str,
    pat_id: PatientID,
    region: str):
    for test_fold in range(5):
        df = load_loader_manifest(datasets, region, test_fold=test_fold)
        df = df[df.loader == 'test']
        df = df[(df['origin-dataset'] == dataset) & (df['origin-patient-id'] == str(pat_id))]
        if len(df) == 1:
            return test_fold

    raise ValueError(f"Patient '{pat_id}' not found for region '{region}' loader and dataset '{dataset}'.")

def get_multi_loader_manifest(
    dataset: Union[str, List[str]],
    check_processed: bool = True,
    n_folds: Optional[int] = 5,
    n_train: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    datasets = arg_to_list(dataset, str)

    # Create empty dataframe.
    cols = {
        'loader': str,
        'loader-batch': int,
        'dataset': str,
        'sample-id': str,
        'origin-dataset': str,
        'origin-patient-id': str
    }
    df = pd.DataFrame(columns=cols.keys())

    # Cache datasets in memory.
    dataset_map = dict((d, ds.get(d, 'training', check_processed=check_processed)) for d in datasets)

    # Create test loader.
    # Create loaders.
    tl, vl, tsl = MultiLoader.build_loaders(datasets, check_processed=check_processed, load_data=False, load_test_origin=False, n_folds=n_folds, n_train=n_train, shuffle_train=False, test_fold=test_fold)
    loader_names = ['train', 'validate', 'test']

    # Get values for this region.
    for loader_name, loader in zip(loader_names, (tl, vl, tsl)):
        for b, pat_desc_b in tqdm(enumerate(iter(loader))):
            for pat_desc in pat_desc_b:
                dataset, sample_id = pat_desc.split(':')
                origin_ds, origin_pat_id = dataset_map[dataset].sample(sample_id).origin
                data = {
                    'loader': loader_name,
                    'loader-batch': b,
                    'dataset': dataset,
                    'sample-id': sample_id,
                    'origin-dataset': origin_ds,
                    'origin-patient-id': origin_pat_id
                }
                df = append_row(df, data)

    # Set type.
    df = df.astype(cols)

    return df

def create_multi_loader_manifest(
    dataset: Union[str, List[str]],
    check_processed: bool = True,
    n_folds: Optional[int] = 5,
    test_fold: Optional[int] = None) -> None:
    datasets = arg_to_list(dataset, str)
    arg_log('Creating multi-loader manifest', ('dataset', 'check_processed', 'n_folds', 'test_fold'), (dataset, check_processed, n_folds, test_fold))

    # Get manifest.
    df = get_multi_loader_manifest(datasets, check_processed=check_processed, n_folds=n_folds, test_fold=test_fold)

    # Save manifest.
    save_csv(df, 'multi-loader-manifests', encode(datasets), f'fold-{test_fold}.csv', overwrite=True)

def load_multi_loader_manifest(
    dataset: Union[str, List[str]],
    test_fold: Optional[int] = None) -> pd.DataFrame:
    datasets = arg_to_list(dataset, str)
    df = load_csv('multi-loader-manifests', encode(datasets), f'fold-{test_fold}.csv')
    df = df.astype({ 'origin-patient-id': str, 'sample-id': str })
    return df

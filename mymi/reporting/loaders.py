import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Union

from mymi import config
from mymi.loaders import Loader
from mymi import logging
from mymi.utils import append_row, encode

def create_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> None:
    if type(datasets) == str:
        datasets = [datasets]
    logging.info(f"Creating loader manifest for datasets '{datasets}', region '{region}', num_folds '{num_folds}', test_fold '{test_fold}'.")

    # Create loaders.
    df = Loader.manifest(datasets, region, num_folds=num_folds, test_fold=test_fold)

    # Save manifest.
    config.save_csv(df, 'loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv', index=False, overwrite=True)

def load_loader_manifest(
    datasets: Union[str, List[str]],
    region: str,
    num_folds: Optional[int] = None,
    test_fold: Optional[int] = None) -> pd.DataFrame:
    return config.load_csv('loader-manifests', encode(datasets), f'{region}-fold-{test_fold}.csv')

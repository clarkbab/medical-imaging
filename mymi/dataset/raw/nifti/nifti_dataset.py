import os
import pandas as pd
from tqdm import tqdm
from typing import List

from mymi import cache
from mymi import config
from mymi import logging
from mymi import regions
from mymi import types

from ...dataset import Dataset, DatasetType
from .nifti_object import NIFTIObject

class NIFTIDataset(Dataset):
    def __init__(
        self,
        name: str):
        self._name = name
        self._path = os.path.join(config.directories.datasets, 'raw', name)
    
    @property
    def description(self) -> str:
        return f"NIFTI: {self._name}"

    @property
    def type(self) -> DatasetType:
        return DatasetType.NIFTI

    def list_ids(self) -> List[str]:
        """
        returns: a list of NIFTI IDs.
        """
        ct_path = os.path.join(self._path, 'raw', 'ct')
        files = os.listdir(ct_path)
        files = [f.replace('.nii.gz', '') for f in files]
        return files

    @cache.method('_name')
    def ct_summary(
        self,
        clear_cache: bool = False) -> pd.DataFrame:
        """
        returns: a DataFrame with patient CT summaries.
        """
        # Define table structure.
        cols = {
            'id': str,
            'fov-x': float,
            'fov-y': float,
            'fov-z': float,
            'hu-max': float,
            'hu-min': float,
            'offset-x': float,
            'offset-y': float,
            'offset-z': float,
            'size-x': int,
            'size-y': int,
            'size-z': int,
            'spacing-x': float,
            'spacing-y': float,
            'spacing-z': float,
        }
        df = pd.DataFrame(columns=cols.keys())

        # List patients.
        ids = self.list_ids()

        # Add patient info.
        logging.info(f"Loading CT summary for dataset '{self._name}'..")
        for id in tqdm(ids):
            # Load object CT info.
            ct_df = self.object(id).ct_summary()

            # Add row.
            ct_df['id'] = id
            df = df.append(ct_df)

        # Set column type.
        df = df.astype(cols)

        return df

    # @cache.method('_name')
    def region_names(self) -> pd.DataFrame:
        """
        returns: a DataFrame with patient region names.
        """
        # Define table structure.
        cols = {
            'id': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        ids = self.list_ids()

        # Add object regions.
        logging.info(f"Loading region names for dataset '{self._name}'..")
        for id in tqdm(ids):
            # Load patient regions.
            names_df = self.object(id).region_names()

            # Add rows.
            for _, row in names_df.iterrows():
                data = {
                    'id': id,
                    'region': row.region,
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    def object(
        self,
        id: str) -> NIFTIObject:
        return NIFTIObject(self._name, id)
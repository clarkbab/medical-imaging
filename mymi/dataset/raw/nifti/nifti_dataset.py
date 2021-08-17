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
from .nifti_patient import NIFTIPatient

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
            # Load patient CT info.
            ct_df = self.patient(id).ct_summary()

            # Add row.
            ct_df['id'] = id
            df = df.append(ct_df)

        # Set column type.
        df = df.astype(cols)

        return df

    # @cache.method('_name')
    def list_regions(self) -> pd.DataFrame:
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

        # Add patient regions.
        logging.info(f"Loading region names for dataset '{self._name}'..")
        for id in tqdm(ids):
            for region in self.patient(id).list_regions():
                data = {
                    'id': id,
                    'region': region,
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df
    
    @cache.method('_name')
    def region_summary(
        self, 
        clear_cache: bool = False,
        regions: types.PatientRegions = 'all') -> pd.DataFrame:
        """
        returns: a DataFrame with patient regions and information.
        kwargs:
            clear_cache: force the cache to clear.
            regions: include patients with (at least) on of the requested regions.
        """
        # Define table structure.
        cols = {
            'id': str,
            'region': str,
            'centroid-mm-x': float,
            'centroid-mm-y': float,
            'centroid-mm-z': float,
            'centroid-voxels-x': float,
            'centroid-voxels-y': float,
            'centroid-voxels-z': float,
            'width-mm-x': float,
            'width-mm-y': float,
            'width-mm-z': float,
            'width-voxels-x': float,
            'width-voxels-y': float,
            'width-voxels-z': float,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each object.
        pats = self.list_patients()

        # Keep patients with (at least) one requested region.
        pats = list(filter(lambda i: self.object(i).has_one_region(regions), pats))

        # Add patient regions.
        logging.info(f"Adding patient region summaries for dataset '{self._name}'..")
        for pat in tqdm(pats):
            # Load object summary.
            summary_df = self.patient(pat).region_summary(clear_cache=clear_cache, regions=regions)

            # Add rows.
            for _, row in summary_df.iterrows():
                data = {
                    'patient-id': pat,
                    'region': row.region,
                    'width-mm-x': row['width-mm-x'],
                    'width-mm-y': row['width-mm-y'],
                    'width-mm-z': row['width-mm-z'],
                    'centroid-mm-x': row['centroid-mm-x'],
                    'centroid-mm-y': row['centroid-mm-y'],
                    'centroid-mm-z': row['centroid-mm-z'],
                    'centroid-voxels-x': row['centroid-voxels-x'],
                    'centroid-voxels-y': row['centroid-voxels-y'],
                    'centroid-voxels-z': row['centroid-voxels-z'],
                    'width-mm-x': row['width-mm-x'],
                    'width-mm-y': row['width-mm-y'],
                    'width-mm-z': row['width-mm-z'],
                    'width-voxels-x': row['width-voxels-x'],
                    'width-voxels-y': row['width-voxels-y'],
                    'width-voxels-z': row['width-voxels-z'],
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    def patient(
        self,
        id: str) -> NIFTIPatient:
        return NIFTIPatient(self._name, id)

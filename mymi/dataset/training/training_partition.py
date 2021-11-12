import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Callable, List

from mymi import logging
from mymi import types

from .training_sample import TrainingSample

class TrainingPartition:
    def __init__(
        self,
        dataset: 'TrainingDataset',
        name: str):
        """
        args:
            dataset: the dataset name.
            name: the partition name.
        """
        self._global_id = f"{dataset}, Partition: {name}"
        self._dataset = dataset
        self._name = name
        self._path = os.path.join(dataset.path, 'data', name)

        # Check if dataset exists.
        if not os.path.exists(self._path):
            raise ValueError(f"Partition '{name}' not found for dataset '{dataset.name}'.")

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def description(self) -> str:
        return self._global_id

    def __str__(self) -> str:
        return self._global_id

    @property
    def manifest(self) -> pd.DataFrame:
        df = self._dataset.manifest()
        df = df[df['partition'] == self._name].drop(columns='partition')
        return df

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def description(self) -> str:
        return f"Partition '{self._name}' of {self.dataset.description}"

    def patient_id(
        self,
        sample_idx: int) -> types.PatientID:
        df = self.manifest
        result_df = df[df['index'] == sample_idx]
        if len(result_df) == 0:
            raise ValueError(f"Sample '{sample_idx}' not found for partition '{self}'.")
        pat_id = result_df['patient-id'].iloc[0] 
        return pat_id

    def list_samples(
        self,
        regions: types.PatientRegions = 'all') -> List[int]:
        """
        returns: the sample indices.
        """
        path = os.path.join(self._path, 'inputs')
        if os.path.exists(path):
            indices = [int(f.replace('.npz', '')) for f in sorted(os.listdir(path))]

            # Filter on sample regions.
            indices = list(filter(self._filter_sample_by_regions(regions), indices))
        else:
            indices = []
        return indices

    def list_regions(self) -> pd.DataFrame:
        """
        returns: a DataFrame with partition region names.
        """
        # Define table structure.
        cols = {
            'sample-index': str,
            'region': str,
        }
        df = pd.DataFrame(columns=cols.keys())

        # Load each patient.
        indices = self.list_samples()

        # Add patient regions.
        logging.info(f"Loading regions for dataset '{self._dataset.name}', partition '{self._name}'...")
        for index in tqdm(indices):
            for region in self.sample(index).list_regions():
                data = {
                    'sample-index': index,
                    'region': region,
                }
                df = df.append(data, ignore_index=True)

        # Set column types.
        df = df.astype(cols)

        return df

    def sample(
        self,
        index: int) -> TrainingSample:
        """
        returns: the partition sample.
        """
        return TrainingSample(self, index)

    def create_input(
        self,
        id: str,
        data: np.ndarray) -> int:
        """
        effect: creates an input sample.
        returns: the index of the sample.
        args:
            id: the object ID.
            data: the data to save.
        """
        # Get next index.
        path = os.path.join(self._path, 'inputs')
        if os.path.exists(path):
            inputs = os.listdir(path)
            if len(inputs) == 0:
                index = -1
            else:
                index = int(list(sorted(inputs))[-1].replace('.npz', '')) + 1
        else:
            os.makedirs(path)
            index = 0

        # Save the input data.
        filename = f'{index}.npz'
        filepath = os.path.join(path, filename)
        np.savez_compressed(filepath, data=data)

        # Update the manifest.
        self._dataset.append_to_manifest(self._name, index, id)

        return index

    def create_label(
        self,
        index: int,
        region: str,
        data: np.ndarray) -> None:
        """
        effect: creates an input sample.
        args:
            pat_id: the patient ID to add to the manifest.
            region: the region name.
            data: the label data.
        """
        # Create region partition if it doesn't exist.
        region_path = os.path.join(self._path, 'labels', region)
        if not os.path.exists(region_path):
            os.makedirs(region_path)

        # Save label.
        filepath = os.path.join(region_path, f'{index}.npz')
        f = open(filepath, 'wb')
        np.savez_compressed(f, data=data)

    def input_summary(self) -> pd.DataFrame:
        cols = {
            'size-x': int,
            'size-y': int,
            'size-z': int
        }
        df = pd.DataFrame(columns=cols.keys())

        for sam_id in tqdm(self.list_samples()):
            row = self.sample(sam_id).input_summary()
            data = {
                'size-x': row['size-x'],
                'size-y': row['size-y'],
                'size-z': row['size-z']
            }
            df = df.append(data, ignore_index=True)

        df = df.astype(cols)
        return df

    def label_summary(
        self,
        regions: types.PatientRegions = 'all') -> pd.DataFrame:
        cols = {
            'sample': int,
            'region': str,
            'size-x': int,
            'size-y': int,
            'size-z': int
        }
        df = pd.DataFrame(columns=cols.keys())

        for sam_id in tqdm(self.list_samples()):
            summary = self.sample(sam_id).label_summary(regions=regions)
            for i, row in summary.iterrows():
                data = {
                    'sample-id': sam_id,
                    'region': row['region'],
                    'size-x': row['size-x'],
                    'size-y': row['size-y'],
                    'size-z': row['size-z']
                }
                df = df.append(data, ignore_index=True)

        df = df.astype(cols)
        return df

    def _filter_sample_by_regions(
        self,
        regions: types.PatientRegions) -> Callable[[str], bool]:
        """
        returns: a function that filters patients on region presence.
        args:
            regions: the passed 'regions' kwarg.
        """
        def fn(index):
            if type(regions) == str:
                if regions == 'all':
                    return True
                else:
                    return self.sample(index).has_region(regions)
            else:
                sam_regions = self.sample(index).list_regions()
                if len(np.intersect1d(regions, sam_regions)) != 0:
                    return True
                else:
                    return False
        return fn

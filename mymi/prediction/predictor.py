import numpy as np
import os

import torch
import torchio
from torch.cuda.amp import autocast
from torchio import ScalarImage, Subject
from torchio.transforms import Compose, CropOrPad, Resample
from tqdm import tqdm
from typing import *
import sys

from mymi import config
from mymi import dataset
from mymi.utils import filterOnPatIDs

sys.path.append('/home/baclark/code/rt-utils')
from rt_utils import RTStructBuilder

class Predictor:
    def __init__(
        self,
        dataset: str,
        network_size: Tuple[int, int, int],
        network_spacing: Tuple[float, float, float],
        device: torch.device = torch.device('cpu'),
        mixed_precision: bool = True,
        pat_ids: Union[str, Sequence[str]] = 'all'):
        """
        args:
            dataset: the dataset to predict.
            network_size: the size expected by the network input layer.
            network_spacing: the spacing expected by the network input layer.
            size: the input size expected by the network.
        kwargs:
            device: the device to use.
            mixed_precision: use mixed precision.
            pat_ids: the patients to predict. 
        """ 
        self._dataset = dataset
        self._device = device
        self._mixed_precision = mixed_precision
        self._network_size = network_size
        self._network_spacing = network_spacing
        self._pat_ids = pat_ids

    def __call__(
        self,
        model: torch.nn.Module) -> None:
        """
        args:
            model: the model to make predictions with.
        """
        # Load the patients.
        dataset.select(self._dataset)
        pats = dataset.list_patients()

        # See if all 'pat_ids' are in the dataset.
        if not np.in1d(self._pat_ids, pats).all():
            raise ValueError(f"Specified patient IDs '{self._pat_ids}', but not all present in dataset '{self._dataset}'.")
        
        # Select specified patients. 
        pats = list(filter(filterOnPatIDs(self._pat_ids), pats))

        # Predict for each patient.
        for pat in tqdm(pats):
            # Load patient summary.
            summary = dataset.patient(pat).ct_summary().iloc[0].to_dict()
            initial_spacing = (summary['spacing-x'], summary['spacing-y'], summary['spacing-z'])
            print('initial spacing: ', initial_spacing)
            print('network spacing: ', self._network_spacing)
            
            # Load patient data.
            input = dataset.patient(pat).ct_data()
            initial_size = input.shape
            input = np.expand_dims(input, axis=0)   # Add batch dim.
            print('loaded size: ', initial_size)

            # Resample to network input layer voxel spacing.
            input = self._resample(input, self._network_spacing, initial_spacing)
            print('with network spacing: ', input.shape)
            
            # Reshape to network input layer size.
            input = self._crop_or_pad(input, self._network_size)
            print('with network size: ', input.shape)

            # Prepare for network.
            input = input.unsqueeze(1)      # Add channel dim.
            input = input.float()
            input = input.to(self._device)

            # Make prediction.
            with autocast(enabled=self._mixed_precision):
                pred = model(input)

            # Get binary prediction.
            pred = pred.argmax(axis=1)

            # Move data to CPU.
            pred = pred.cpu()
            print('pred size: ', pred.shape)

            # Resample to CT spacing.
            pred = self._resample(pred, initial_spacing, self._network_spacing)
            print('with restored spacing: ', pred.shape) 

            # Resize to original size - accounting for rounding errors.
            pred = self._crop_or_pad(pred, initial_size)
            print('with restored size: ', pred.shape)

            # Inspect pred.
            print('pred type: ', type(pred))
            print('pred dtype: ', pred.dtype)

            # Convert to numpy ndarray.
            pred = pred.numpy()
            pred = pred.astype(bool)
            pred = pred.squeeze(0)      # Remove batch dim.

            # Create/save RTSTRUCT.
            dicom_path = os.path.join(config.directories.datasets, self._dataset, 'hierarchical', pat, 'ct')
            rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_path)
            rtstruct.add_roi(mask=pred, name='Parotid-Left')
            new_dataset = f"{self._dataset}-pred"
            rtstruct_path = os.path.join(config.directories.datasets, new_dataset, 'hierarchical', pat, 'rtstruct', 'pred.dcm')
            os.makedirs(os.path.dirname(rtstruct_path), exist_ok=True)
            rtstruct.save(rtstruct_path)

    def _resample(
        self,
        array: np.ndarray,
        new_spacing: Tuple[float, float, float],
        spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        returns: an ndarray with new size/spacing.
        args:
            array: the array to resample.
            new_spacing: the new spacing.
            spacing: the current spacing.
        """
        transform = Compose([
            Resample(new_spacing)
        ])
        affine = np.array([
            [spacing[0], 0, 0, 0],
            [0, spacing[1], 0, 0],
            [0, 0, spacing[2], 1],
            [0, 0, 0, 1]
        ])
        array = ScalarImage(tensor=array, affine=affine)
        subject = Subject(input=array)
        output = transform(subject)
        array = output['input'].data

        return array

    def _crop_or_pad(
        self,
        array: np.ndarray,
        new_size: Tuple[int, int, int]) -> np.ndarray:
        """
        returns: a cropped/padded ndarray.
        args:
            array: the array to resize.
            new_size: the new size.
        """
        transform = Compose([
            CropOrPad(new_size, padding_mode='minimum')
        ])
        affine = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])
        array = ScalarImage(tensor=array, affine=affine)
        subject = Subject(input=array)
        output = transform(subject)
        array = output['input'].data

        return array

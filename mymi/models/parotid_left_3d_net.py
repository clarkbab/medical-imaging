import numpy as np
import torch
import torch.nn as nn
from torchio import ScalarImage, Subject
from torchio.transforms import CropOrPad, Resample
from typing import Any, Tuple

class ParotidLeft3DNet(nn.Module):
    def __init__(
        self,
        localiser: nn.Module,
        segmenter: nn.Module,
        localiser_size: Tuple[int, int, int],
        localiser_spacing: Tuple[float, float, float]):
        """
        effect: initialises the network.
        args:
            localiser: the localisation module.
            segmenter: the segmentation module.
        """
        super().__init__()

        self._localiser = localiser
        self._segmenter = segmenter
        self._spacing = (1, 1, 3)
        self._localiser_size = (128, 128, 96)
        self._localiser_spacing = (4, 4, 6.625)
        self._segmenter_size = (128, 128, 96)

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        """
        returns: the inference result.
        args:
            x: the batch of input volumes. Voxel spacing should be (1, 1, 3).
        """
        print('input shape:', x.shape)

        # Get predicted OAR location.
        pred = self._get_location(x)
        print('localiser pred shape:', pred.shape)

        # Get segmentation prediction.
        pred = self._get_segmentation(x, pred)
        print('segmenter pred shape:', pred.shape)

        return pred
    
    def _get_location(
        self,
        x: torch.Tensor) -> torch.Tensor:
        """
        returns: a 3D binary array with location prediction at (1, 1, 3) spacing.
        args:
            x: the input 3D array at (1, 1, 3) spacing.
        """
        # Get device.
        device = x.device

        # Create downsampled input.
        if device.type == 'cuda':
            x = x.cpu()
        x = self._resample(x, self._spacing, self._localiser_spacing)

        # Save resampled size. We need to crop/pad our localiser prediction to it's original shape
        # before resampling to attain the correct full-resolution shape.
        x_size = x.shape

        # Create cropped/padded input.
        x = self._crop_or_pad(x, self._localiser_size)

        # Get localiser result.
        x = x.unsqueeze(1)       # Add required 'channel' dimension.
        if device.type == 'cuda':
            x = x.cuda()
        pred = self._localiser(x)

        # Get binary mask.
        pred = pred.argmax(axis=1)

        # Reverse the crop/pad.
        if device.type == 'cuda':
            pred = pred.cpu()
        pred = self._crop_or_pad(pred, x_size)

        # Upsample to full resolution.
        pred = self._resample(pred, self._localiser_spacing, self._spacing)
        if device.type == 'cuda':
            pred = pred.gpu()

        return pred

    def _get_segmentation(
        self,
        x: torch.Tensor,
        pred: torch.Tensor) -> torch.Tensor:
        """
        returns: the full result of segmentation at (1, 1, 3) spacing.
        args:
            x: the input tensor at (1, 1, 3) spacing.
            pred: the location prediction at (1, 1, 3) spacing.
        """
        # Get device.
        device = x.device

        # Extract patch around bounding box.
        print('segmentation input shape:', x.shape)
        if device == 'cuda':
            x = x.cpu()
            pred = pred.cpu()
        x, crop_or_padding = self._extract_patch(x, pred, self._segmenter_size)
        print('patch shape:', x.shape)
        print('crop or padding:', crop_or_padding)

        # Pass patch to segmenter.
        x = x.unsqueeze(1)       # Add required 'channel' dimension.
        if device == 'cuda':
            x = x.cuda()
        pred = self._segmenter(x)
        print('segmentation shape:', pred.shape)

        # Pad segmentation prediction.
        if device == 'cuda':
            pred = pred.cpu()
        crop_or_padding = tuple((-d[0], -d[1]) for d in crop_or_padding)    # Reverse crop/padding amounts.
        print('reverse crop or padding:', crop_or_padding)
        pred = self._asymmetric_crop_or_pad(pred, crop_or_padding)
        print('asymmetric crop or pad:', pred.shape)

        return pred

    def _resample(
        self,
        x: torch.Tensor,
        before: Tuple[float, float, float],
        after: Tuple[float, float, float]) -> torch.Tensor:
        """
        returns: a resampled tensor.
        args:
            x: the data to resample.
            before: the spacing before resampling.
            after: the spacing after resampling.
        """
        # Create the transform.
        transform = Resample(after)

        # Create 'subject'.
        affine = np.array([
            [before[0], 0, 0, 0],
            [0, before[1], 0, 0],
            [0, 0, before[2], 1],
            [0, 0, 0, 1]
        ])
        x = ScalarImage(tensor=x, affine=affine)
        subject = Subject(input=x)

        # Transform the subject.
        output = transform(subject)

        # Extract results.
        x = output['input'].data

        return x

    def _crop_or_pad(
        self,
        x: torch.Tensor,
        after: Tuple[int, int, int]) -> torch.Tensor:
        """
        returns: a cropped/padded ndarray.
        args:
            x: the tensor to resize.
            after: the new size.
        """
        # Create transform.
        transform = CropOrPad(after, padding_mode='minimum')

        # Create subject.
        affine = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1]
        ])
        x = ScalarImage(tensor=x, affine=affine)
        subject = Subject(x=x)

        # Perform transformation.
        output = transform(subject)

        # Get result.
        x = output['x'].data

        return x

    def _extract_patch(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        size: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """
        returns: a (patch, crop_or_padding) tuple where the patch is the extracted patch centred around the OAR,
            and the crop_or_padding tells us how much was added/removed at each end of each dimension.
        args:
            x: the input data.
            pred: the label data.
            size: the patch will be this size with the OAR in the centre. Must be larger than OAR extent.
        raises:
            ValueError: if the OAR extent is larger than the patch size.
        """
        # Find OAR extent.
        non_zero = np.argwhere(pred != 0)
        mins = non_zero.min(axis=0)
        maxs = non_zero.max(axis=0)
        oar_size = maxs - mins

        # Check oar size.        
        if (oar_size > extent).any():
            raise ValueError(f"OAR size '{oar_size}' larger than requested patch size '{size}'.")

        # Determine min/max indices of the patch.
        size_diff = size - oar_size
        lower_add = np.ceil(size_diff / 2).astype(int)
        mins = mins - lower_add
        maxs = mins + size

        # Check for negative indices, and record padding.
        lower_pad = (-mins).clip(0) 
        mins = mins.clip(0)

        # Check for max indices larger than input size, and record padding.
        upper_pad = (maxs - x.shape).clip(0)
        maxs = maxs - upper_pad

        # Perform crop.
        slices = tuple(slice(min, max) for min, max in zip(mins, maxs))
        x = x[slices]

        # Perform padding.
        padding = tuple(zip(lower_pad, upper_pad))
        x = np.pad(x, padding, padding_mode='minimum')

        # Get crop or padding information.
        info = tuple(zip(mins, maxs - x.shape))

        return x, info

    def _asymmetric_crop_or_pad(
        self,
        x: torch.Tensor,
        crop_or_padding: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> torch.Tensor:
        """
        returns: a 3D array with dimensions cropped or padded.
        args:
            x: the input tensor.
            crop_or_padding: number of voxels to add remove from each dimension.
        """
        # Perform padding.
        padding = np.array(crop_or_padding).clip(0)
        x = np.pad(x, padding)

        # Perform cropping.
        cropping = (-np.array(crop_or_padding)).clip(0)
        mins = tuple(d[0] for d in cropping)
        maxs = tuple(s - d[1] for d, s in zip(cropping, x.shape))
        slices = tuple(slice(min, max) for min, max in zip(mins, maxs))
        x = x[slices]

        return x


import torch.nn as nn

class ParotidLeft3DNet(nn.Module):
    def __init__(
        self,
        localiser: nn.Module,
        segmenter: nn.Module,
        spacing: Tuple[float, float, float],
        localiser_size: Tuple[int, int, int],
        localiser_spacing: Tuple[float, float, float]):
        """
        effect: initialises the network.
        args:
            localiser: the localisation module.
            segmenter: the segmentation module.
            spacing: spacing of input data.
            localiser_size: the input size required by the localiser.
            localiser_spacing: the spacing expected by the localiser.
        """
        super().__init__()

        self._localiser = localiser
        self._segmenter = segmenter
        self._spacing = spacing
        self._localiser_size = localiser_size
        self._localiser_spacing = localiser_spacing

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        """
        returns: the inference result.
        args:
            x: the batch of input volumes.
        """
        # Create downsampled input.
        x_loc = self._resample(x, self._spacing, self._localiser_spacing)

        # Save resampled size. We need to resize our localiser prediction to it's original shape
        # before resampling to attain the correct full-resolution shape.
        x_loc_size = x_loc.shape

        # Create cropped/padded input.
        x_loc = self._crop_or_pad(x_loc, self._localiser_size)

        # Get localiser result.
        pred_loc = self._localiser(x_loc)

        # Get binary mask.
        pred_loc = pred_loc.argmax(axis=1)

        # Reverse the crop/pad.
        pred_loc = self._crop_or_pad(pred_loc, x_loc_size)

        # Upsample to full resolution.
        pred_loc = self._resample(pred_loc, self._localiser_spacing, self._spacing)

        # Extract patch around bounding box.
        x_patch, crop_or_padding = self._extract_patch(x, pred_loc)

        # Pass patch to segmenter.
        pred_seg = self._segmenter(x_patch)

        # Pad segmentation prediction.
        pred_seg = pred_seg

        return pred_seg

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
        x = output['input'].data.squeeze(0)

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
        size: Tuple[int, int, int]) -> Tuple[torch.Tensor, Any]:
        """
        returns: a patch around the OAR.
        args:
            x: the input data.
            pred: the label data.
            size: the size of the patch. Must be larger than the extent of the OAR.
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

        return x

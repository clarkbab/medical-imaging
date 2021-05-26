import torch.nn as nn

class ParotidLeft3DNet(nn.Module):
    def __init__(
        self,
        input_spacing: Tuple[float, float, float],
        localiser: nn.Module,
        localiser_spacing: Tuple[float, float, float],
        segmenter: nn.Module):
        """
        effect: initialises the network.
        args:
            input_spacing: input to 'forward' assumed to have this spacing.
            localiser: the localisation module.
            localiser_spacing: the spacing expected by the localiser.
            segmenter: the segmentation module.
        """
        super().__init__()

        self._input_spacing = input_spacing
        self._localiser = localiser
        self._localiser_spacing = localiser_spacing
        self._segmenter = segmenter

    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        """
        returns: the inference result.
        args:
            x: the batch of input volumes.
        """
        # Create downsampled input.
        x_down = self._resample_for_localiser(x)

        # Get localiser result.
        pred_loc = self._localiser(x_down)

        # Upsample to full resolution.
        pred_loc_up = self._upsample_prediction(pred_loc)

        # Find bounding box.
        # ((x_min, x_max), (y_min, y_max), ...)
        oar_bb = self._get_bounding_box(pred_loc_up)

        # Extract patch around bounding box.
        shape = (128, 128, 96)
        x_patch = self._extract_patch(x, oar_bb, shape)

        # Pass patch to segmenter.
        pred_seg = self._segmenter(x_patch)

        # Pad segmentation prediction.
        pred_seg = pred_seg

        return pred_seg

    def _resample_for_localiser(
        self,
        x: torch.Tensor) -> torch.Tensor:
        """
        returns: a resampled tensor.
        args:
            x: the data to resample.
        """
        # Create the transform.
        transform = Resample(self._localiser_spacing)

        # Add 'batch' dimension.
        x = x.unsqueeze(0)

        # Create 'subject'.
        affine = np.array([
            [self._input_spacing[0], 0, 0, 0],
            [0, self._input_spacing[1], 0, 0],
            [0, 0, self._input_spacing[2], 1],
            [0, 0, 0, 1]
        ])
        x = ScalarImage(tensor=x, affine=affine)
        subject = Subject(one_image=x)

        # Transform the subject.
        output = transform(subject)

        # Extract results.
        x = output['one_image'].data.squeeze(0)

        return x

    def _extract_oar_patch(
        self,
        input: torch.Tensor,
        label: torch.Tensor,
        shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        returns: a patch around the OAR.
        args:
            input: the input data.
            label: the label data.
            shape: the shape of the patch.
        """
        # Required extent.
        # From 'Segmenter dataloader' notebook, max extent in training data is (48.85mm, 61.52mm, 72.00mm).
        # Converting to voxel width we have: (48.85, 61.52, 24) for a spacing of (1.0mm, 1.0mm, 3.0mm).
        # We can choose a patch that is larger than the required voxel width, and that we know fits into the GPU
        # as we use it for the localiser training: (128, 128, 96), giving physical size of (128mm, 128mm, 288mm)
        # which is more than large enough. We can probably trim this later.

        # Find OAR extent.
        non_zero = np.argwhere(label != 0)
        mins = non_zero.min(axis=0)
        maxs = non_zero.max(axis=0)
        oar_shape = maxs - mins

        # Pad the OAR to the required shape.
        shape_diff = shape - oar_shape
        lower_add = np.ceil(shape_diff / 2).astype(int)
        mins = mins - lower_add
        maxs = mins + shape

        # Crop or pad the volume.
        input = self._crop_or_pad(input, mins, maxs, fill=input.min()) 
        label = self._crop_or_pad(label, mins, maxs)

        return input, label


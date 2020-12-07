import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F

class CropOrPad:
    def __init__(self, resolution):
        """
        resolution: the desired image resolution.
        padding: add padding when cropped image is less than desired resolution.
        """
        self.resolution = resolution

    def __call__(self, input, label):
        """
        returns: crops or pads the sample data ensuring that the desired resolution is achieved.
        input: the input data.
        label: the label data.
        """
        input, label = sample[0], sample[1]

        # Use Pytorch crop.
        y, x, h, w = transforms.RandomCrop.get_params(input, self.resolution)
        input = F.crop(input, y, x, h, w, pad_if_needed=True)
        label = F.crop(label, y, x, h, w)
        return input, label

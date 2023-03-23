import torchio as tio
import torch
import numpy as np

class Standardise(tio.Transform):
    def __init__(self, mean: float, std: float):
        super().__init__()
        self.__mean = mean
        self.__std = std

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        image = subject['input']
        np_data = image.data.squeeze().numpy()
        np_data = (np_data - self.__mean) / self.__std
        image.data = torch.from_numpy(np_data).unsqueeze(0)
        return subject

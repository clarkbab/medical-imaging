import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from augmed import Transform
from dicomset.training import TrainingDataset
from mymi.typing import *
from mymi.utils.io import load_numpy

def normalise_points(
    points: np.ndarray,
    min: float = 0,
    max: float = 1,
) -> np.ndarray:
    norm_points = []
    for p in points:
        # y-axis only.
        y_min, y_max = p[:, 1].min(), p[:, 1].max()
        p[:, 1] = (max - min) * (p[:, 1] - y_min) / (y_max - y_min) + min
        norm_points.append(p)
    return np.stack(norm_points, axis=0)

class BSPLoader:
    @staticmethod
    def build_loaders(
        dataset: str,
        batch_size: int = 1,    # Shrouds have different x-size, we could look at clever batching methods later.
        num_workers: int = 0,
        transform_train: Transform | None = None,
        transform_val: Transform | None = None,
        ) -> Tuple[DataLoader, DataLoader]:
        set = TrainingDataset(dataset)

        train_set = TrainingSet(set, transform=transform_train)
        val_set = ValidationSet(set, transform=transform_val)

        train_loader = DataLoader(batch_size=batch_size, dataset=train_set, num_workers=num_workers, shuffle=True)
        val_loader = DataLoader(batch_size=batch_size, dataset=val_set, num_workers=num_workers, shuffle=False)

        return train_loader, val_loader

class TrainingSet(Dataset):
    def __init__(
        self,
        dataset: TrainingDataset,
        transform: Transform | None = None,
    ) -> None:
        self.__transform = transform
        dirpath = os.path.join(dataset.path, 'data', 'train')
        self.__filepaths = sorted([
            os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.npz')
        ])

    def __len__(self) -> int:
        return len(self.__filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Signals are already normalise to [0, 1].
        filepath = self.__filepaths[idx]
        shroud, signals = load_numpy(filepath, keys=['shroud', 'signals'])
        assert shroud.shape[0] == signals.shape[1], f"Shroud frames ({shroud.shape}) and signal N points ({signals.shape}) must match."

        if self.__transform is not None:
            # Normalise signals to the shrould image for augmentation as landmarks.
            y_size = shroud.shape[1]
            signals = normalise_points(signals, min=0, max=y_size - 1)

            # Perform transforms.
            # AugMed doesn't currently support point batches.
            amp, phase = signals
            t = self.__transform.freeze()
            shroud, amp, phase = t(shroud, amp, phase)
            signals = np.stack([amp, phase], axis=0)

            # # Normalise signal y-coordinates to [0, 1] after augmentation.
            signals = normalise_points(signals)

        assert shroud.shape[0] == signals.shape[1], f"Shroud frames ({shroud.shape}) and signal N points ({signals.shape}) must match after transform."

        shroud = torch.from_numpy(shroud)
        signals = torch.from_numpy(signals)

        # Add channel dimension: (1, n_frames, y_size).
        shroud = shroud[None, ...]

        return shroud, signals


class ValidationSet(Dataset):
    def __init__(
        self,
        dataset: TrainingDataset,
        transform: Transform | None = None,
    ) -> None:
        self.__transform = transform
        dirpath = os.path.join(dataset.path, 'data', 'validation')
        self.__filepaths = sorted([
            os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.endswith('.npz')
        ])

    def __len__(self) -> int:
        return len(self.__filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Signals are already normalise to [0, 1].
        shroud, signals = load_numpy(self.__filepaths[idx], keys=['shroud', 'signals'])

        if self.__transform is not None:
            # Normalise signals to the shrould image for augmentation as landmarks.
            y_size = shroud.shape[1]
            signals = normalise_points(signals, min=0, max=y_size - 1)
            amp, phase = signals

            # Perform transforms.
            # AugMed doesn't currently support point batches.
            shroud, amp, phase = self.__transform(shroud, amp, phase)
            signals = np.stack([amp, phase], axis=0)

            # Normalise signal y-coordinates to [0, 1] after augmentation.
            signals = normalise_points(signals)

        shroud = torch.from_numpy(shroud)
        signals = torch.from_numpy(signals)

        # Add channel dimension: (1, n_frames, y_size).
        shroud = shroud[None, ...]

        return shroud, signals

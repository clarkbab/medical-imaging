import torch
from torch.utils.data import Sampler
from typing import Sized

class RandomSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,
        epoch: int = 0,
        random_seed: float = 0):
        super().__init__(data_source)
        self.__epoch = epoch
        self.__n_samples = len(data_source)
        self.__random_seed = random_seed

    def __iter__(self):
        # Create random number generator.
        # Seed is based on 'random_seed' and starting epoch - this allows
        # us to create deterministic training runs, even when training is resumed
        # following a crash.
        seed = self.__random_seed + self.__epoch
        print(f"setting random seed {seed}")
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Shuffle indices using the new generator.
        indices = torch.randperm(self.__n_samples, generator=generator).tolist()

        # Increment epoch.
        self.__epoch += 1

        return iter(indices)

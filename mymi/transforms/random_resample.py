import math
import numpy as np
from skimage import transform

class RandomResample:
    def __init__(self, stretch, p=1.0):
        self.stretch = stretch
        self.p = p

    def __call__(self, input, label):
        if np.random.binomial(1, self.p):
            input, label = self.resample(input, label)

        return input, label

    def resample(self, input, label):
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        input_dtype, label_dtype = input.dtype, label.dtype

        # Sample the stretch 
        rand_stretch = [np.random.uniform(*s) for s in self.stretch]

        # Get new resolution.
        assert input.shape == label.shape
        new_res = [math.floor(s * input.shape[i]) for i, s in enumerate(rand_stretch)]

        # Perform resample.
        input = transform.resize(input, new_res, order=3, preserve_range=True)
        label = transform.resize(label, new_res, order=0, preserve_range=True)

        # Reset types.
        input, label = input.astype(input_dtype), label.astype(label_dtype)

        return input, label

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")

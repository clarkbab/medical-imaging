import math
import numpy as np
from skimage import transform

class RandomResample:
    def __init__(self, range, p=1.0):
        self.range = range
        self.p = p

    def __call__(self, data, binary=False, info=None):
        """
        returns: the input data.
        args:
            data: the input data.
        kwargs:
            binary: indicates that input was binary data.
            info: optional info that the transform may require.
        """
        # Get deterministic function.
        det_fn = self.deterministic()
        
        # Process result.
        result = det_fn(data, binary=binary, info=info) 
        return result

    def deterministic(self):
        """
        returns: a deterministic function with same signature as '__call__'.
        """
        # Realise randomness.
        applied, stretch = self.realise_randomness()

        # Create function that can be called to produce consistent results.
        def fn(data, binary=False, info=None):
            if applied:
                data = self.resample(data, stretch, binary)

            return data

        return fn

    def realise_randomness(self):
        """
        returns: all realisations of random processes in the transformation.
        """
        # Determine if rotation is applied.
        applied = True if np.random.binomial(1, self.p) else False

        # Determine angles.
        stretch = np.random.uniform(*self.range)
        
        return applied, stretch 

    def resample(self, data, stretch, binary):
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        dtype = data.dtype 

        # Get new resolution.
        new_res = [math.floor(stretch * data.shape[i]) for i in range(2)] 

        # Perform resample.
        if binary:
            data = transform.resize(data, new_res, order=0, preserve_range=True)
        else:
            data = transform.resize(data, new_res, order=3, preserve_range=True)

        # Reset types.
        data = data.astype(dtype)

        return data

    def cache_key(self):
        raise ValueError("Random transformations aren't cacheable.")

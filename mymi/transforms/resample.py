import hashlib
import json
import numpy as np
import scipy
from scipy.ndimage import zoom

class Resample:
    def __init__(self, resolution=None, spacing=None):
        """
        kwargs:
            resolution: the desired resolution.
            spacing: a list of desired (x, y, z) spacings.
        """
        assert resolution == None or spacing == None, "Either resolution or spacing must be 'None'."

        self.resolution = resolution
        self.spacing = spacing

    def __call__(self, data, binary=False, info=None):
        """
        returns: resampled data.
        data: the data to resample.
        info: information required by the resampling method.
            order: the order of the interpolation method.
            spacing-x: the spacing between voxels in x-direction.
            spacing-y: the spacing between voxels in y-direction.
            spacing-z: the spacing between voxels in z-direction.
        """
        # Get old shape.
        old_shape = data.shape

        # Get resize factor - multiplier for resolution required by 'zoom'.
        if self.spacing is not None:
            # Get old spacing from info.
            old_spacing = np.array([info['spacing-x'], info['spacing-y'], info['spacing-z']])

            resize_factor = self.spacing_resize_factor(old_shape, old_spacing)
        elif self.resolution is not None:
            resize_factor = self.resolution_resize_factor(old_shape)

        # Perform resampling.
        if binary:
            data = zoom(data, resize_factor, order=0)
        else:
            data = zoom(data, resize_factor, order=3)

        return data

    def deterministic(self):
        """
        returns: a deterministic function with same signature as '__call__'.
        """
        # No randomness, just return function identical to '__call__'.
        def fn(data, binary=False, info=None):
            return self.__call__(data, binary=binary, info=info)

        return fn

    def spacing_resize_factor(self, old_shape, old_spacing): 
        # Calculate shape resize factor - the ratio of old to new pixel spacings.
        resize_factor = old_spacing / self.spacing

        # Calculate new shape - rounded to nearest integer.
        new_shape = np.round(old_shape * resize_factor)

        # 'Real' resize factor will be different from desired resize factor as
        # we must maintain integer values for number of pixels.
        resize_factor = new_shape / old_shape

        return resize_factor

    def resolution_resize_factor(self, old_shape):
        # Calculate ratio of new to old shapes.
        resize_factor = np.array(self.resolution) / old_shape

        return resize_factor

    def cache_key(self):
        """
        returns: an ID that is unique based upon transform parameters.
        """
        params = {
            'spacing': self.spacing
        }
        return hashlib.sha1(json.dumps(params).encode('utf-8')).hexdigest()

import hashlib
import json
import numpy as np
import scipy
from scipy.ndimage import zoom

class ResampleTransform:
    def __init__(self, spacing):
        """
        spacing: a list of desired (x, y, z) spacings.
        """
        self.spacing = spacing

    def cache_key(self):
        """
        returns: an ID that is unique based upon transform parameters.
        """
        params = {
            'spacing': self.spacing
        }
        return hashlib.sha1(json.dumps(params).encode('utf-8')).hexdigest()

    def __call__(self, data, info):
        """
        returns: resampled data.
        data: the data to resample.
        info: information required by the resampling method.
            order: the order of the interpolation method.
            spacing-x: the spacing between voxels in x-direction.
            spacing-y: the spacing between voxels in y-direction.
            spacing-z: the spacing between voxels in z-direction.
        """
        new_spacing = np.array([1.0, 1.0, 3.0])
        old_spacing = np.array([info['spacing-x'], info['spacing-y'], info['spacing-z']])

        # Check if resampling is needed.
        if np.array_equal(new_spacing, old_spacing):
            return data

        # Calculate shape resize factor - the ratio of new to old pixel numbers.
        resize_factor = old_spacing / new_spacing

        # Calculate new shape - rounded to nearest integer.
        new_shape = np.round(data.shape * resize_factor)

        # Our real spacing will be different from 'new spacing' due to shape
        # consisting of integers. The field-of-view (shape * spacing) must be
        # maintained throughout.
        real_resize_factor = new_shape / data.shape
        new_spacing = old_spacing / real_resize_factor

        # Perform resampling.
        order = info['order'] if 'order' in info.keys() else 3

        # TODO: Look into using skimage.transform.resize method.
        # https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
        data = zoom(data, real_resize_factor, order=order)

        return data

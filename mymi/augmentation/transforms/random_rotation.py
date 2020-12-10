import hashlib
import json
import numpy as np
from skimage import transform

class RandomRotation:
    def __init__(self, angle, fill=0, p=1.0):
        """
        angle: a (min, max) tuple describing angle range in degrees.
        fill: values with which to fill new pixels.
        """
        self.angle = angle
        self.fill = fill
        self.p = p

    def cache_id(self):
        """
        returns: an ID that is unique based upon transform parameters.
        """
        params = {
            'angle': self.angle,
            'fill': self.fill
        }
        return hashlib.sha1(json.dumps(params).encode('utf-8')).hexdigest()

    def __call__(self, input, label):
        if np.random.binomial(1, self.p):
            input, label = self.rotate(input, label)

        return input, label

    def rotate(self, input, label):
        """
        returns: the (input, label) pair of rotated images.
        input: the input data.
        label: the label data.
        """
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        input_dtype, label_dtype = input.dtype, label.dtype

        # Sample the range uniformly.
        rand_angle = np.random.uniform(*self.angle)

        # Rotate the data.
        # TODO: Look into where we should crop or not?
        input = transform.rotate(input, rand_angle, cval=self.fill, order=3, preserve_range=True)
        label = transform.rotate(label, rand_angle, cval=0, order=0, preserve_range=True)

        # Reset types.
        input, label = input.astype(input_dtype), label.astype(label_dtype)

        return input, label

import hashlib
import json
import numpy as np
from skimage import transform

class RandomRotation:
    def __init__(self, angle, fill=0):
        """
        angle: a (min, max) tuple describing angle range in degrees.
        fill: values with which to fill new pixels.
        """
        self.angle = angle
        self.fill = fill

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
        """
        returns: the (input, label) pair of rotated images.
        input: the input data.
        label: the label data.
        """
        # Sample the range uniformly.
        rand_angle = np.random.uniform(*self.angle)

        # Rotate the data.
        print(label)
        input = transform.rotate(input, rand_angle, cval=self.fill, preserve_range=True)
        label = transform.rotate(label, rand_angle, preserve_range=True)

        return input, label

import hashlib
import json
import math
import numpy as np

class CropTransform:
    def __init__(self, resolution):
        """
        resolution: the desired image resolution.
        """
        self.resolution = resolution

    def cache_id(self):
        """
        returns: an ID that is unique based upon transform parameters.
        """
        params = {
            'resolution': self.resolution
        }
        return hashlib.sha1(json.dumps(params).encode('utf-8')).hexdigest()

    def run(self, data, info):
        """
        data: the data to transform.
        info: information required by the crop method.
        """
        half_ranges = np.array(self.resolution) / 2
        centres = (np.array(data.shape) / 2).astype(int)
        lower_bounds = [np.max([b, 0]) for b in centres - list(map(math.floor, half_ranges))]
        upper_bounds = centres + list(map(math.ceil, half_ranges))
        ranges = [slice(l, u) for l, u in zip(lower_bounds, upper_bounds)]

        # Perform crop.
        return data[ranges]

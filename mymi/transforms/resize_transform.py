import hashlib
import json
import math
import numpy as np

class ResizeTransform:
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

    def __call__(self, data, info):
        """
        data: the data to transform.
        info: information required by the crop method.
        """
        # Determine which dimensions to reshape.
        resolution = [r if r is not None else d for r, d in zip(self.resolution, data.shape)]

        # Create placeholder array.
        new_data = np.zeros(shape=resolution, dtype=data.dtype)

        # Find data centres as we will perform centred cropping and padding.
        data_centre = (np.array(data.shape) - 1) / 2
        new_data_centre = (np.array(new_data.shape) - 1) / 2

        # Find the write range.
        write_shape = np.minimum(new_data.shape, data.shape)
        write_lower_bound = np.array(list(map(math.ceil, new_data_centre - write_shape / 2)))
        write_range = [slice(l, l + r) for l, r in zip(write_lower_bound, write_shape)]

        # Find the read range.
        read_lower_bound = np.array(list(map(math.ceil, data_centre - write_shape / 2)))
        read_range = [slice(l, l + r) for l, r in zip(read_lower_bound, write_shape)]

        # Add data to placeholder.
        new_data[write_range] = data[read_range]

        return new_data

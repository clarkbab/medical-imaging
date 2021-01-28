import hashlib
import json
import numpy as np
from skimage import transform

class RandomTranslation:
    def __init__(self, translation, fill=0, p=1.0):
        """
        translation: an (x, y) tuple of translation ranges in voxels.
        fill: value to use for new pixels.
        p: the probability that the transform is applied.
        """
        self.translation = translation
        self.fill = fill
        self.p = p

    def __call__(self, input, label):
        if np.random.binomial(1, self.p):
            input, label = self.translate(input, label)
        
        return input, label

    def translate(self, input, label):
        """
        input: the input data.
        label: the label data.
        """
        # Preserve data types - important as some input/label pairs in a batch
        # won't be transformed and need the same data type for collation.
        input_dtype, label_dtype = input.dtype, label.dtype

        # Get translation values.
        # Must reverse sign of number as skimage moves viewport, not image by default.
        rand_trans = -np.array([np.random.uniform(*t) for t in self.translation])

        # Apply transformation.
        # TODO: Look into whether we should crop or not.
        tform = transform.AffineTransform(translation=(rand_trans[1], rand_trans[0]))
        input = transform.warp(input, tform, cval=self.fill, order=3, preserve_range=True)
        label = transform.warp(label, tform, cval=0, order=0, preserve_range=True)

        # Reset types.
        input, label = input.astype(input_dtype), label.astype(label_dtype)

        return input, label
        
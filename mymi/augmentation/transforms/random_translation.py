import numpy as np
from skimage import transform

class RandomTranslation:
    def __init__(self, translation, fill=0):
        """
        translation: an (x, y) tuple of translation ranges.
        fill: value to use for new pixels.
        """
        self.translation = translation
        self.fill = fill

    def __call__(self, input, label):
        """
        input: the input data.
        label: the label data.
        """
        # Get translation values.
        # Must reverse sign of number as skimage moves viewport, not image by default.
        rand_trans = -np.array([np.random.uniform(*t) for t in self.translation])

        # Apply transformation.
        tform = transform.AffineTransform(translation=(rand_trans[1], rand_trans[0]))
        input = transform.warp(input, tform, cval=self.fill, preserve_range=True)
        label = transform.warp(label, tform, preserve_range=True)

        return input, label
        
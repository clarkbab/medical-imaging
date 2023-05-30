import numpy as np

from mymi.transforms import centre_crop_or_pad_3D

def naive_crop(input, labels, spacing=None):
    assert spacing is not None

    # Crop input.
    # crop_mm = (320, 520, 730)   # With 60 mm margin (30 mm either end) for each axis.
    crop_mm = (250, 400, 500)   # With 60 mm margin (30 mm either end) for each axis.
    crop = tuple(np.round(np.array(crop_mm) / spacing).astype(int))
    input = centre_crop_or_pad_3D(input, crop)

    # Crop labels.
    for r in labels.keys():
        labels[r] = centre_crop_or_pad_3D(labels[r], crop)

    return input, labels
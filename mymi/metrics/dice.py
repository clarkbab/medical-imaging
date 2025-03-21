import numpy as np
import SimpleITK as sitk

def dice(
    a: np.ndarray,
    b: np.ndarray) -> float:
    """
    returns: the dice coefficient.
    args:
        a: an X x Y x Z boolean array.
        b: an X x Y x Z boolean array.
    """
    if a.shape != b.shape:
        raise ValueError(f"Metric 'dice' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")

    # 'SimpleITK' filter doesn't handle empty a/b.
    if a.sum() == 0 and b.sum() == 0:
        return 1.0

    # Convert types for SimpleITK.
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    a = sitk.GetImageFromArray(a)
    b = sitk.GetImageFromArray(b)
    filter = sitk.LabelOverlapMeasuresImageFilter()
    filter.Execute(a, b)
    dice = filter.GetDiceCoefficient()
    return dice

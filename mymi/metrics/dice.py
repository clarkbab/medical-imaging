import numpy as np
import SimpleITK as sitk
import torch

def dice(
    a: np.ndarray,
    b: np.ndarray) -> float:
    """
    returns: the dice coefficient.
    args:
        a: a 3D array.
        b: another 3D array.
    """
    if a.shape != b.shape:
        raise ValueError(f"Dice coefficient expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")

    a = sitk.GetImageFromArray(a)
    b = sitk.GetImageFromArray(b)
    filter = sitk.LabelOverlapMeasuresImageFilter()
    filter.Execute(a, b)
    dice = filter.GetDiceCoefficient()
    return dice

def batch_mean_dice(
    a: np.ndarray,
    b: np.ndarray) -> float:
    """
    returns: the mean batch dice coefficient.
    args:
        a: a 4D array.
        b: another 4D array.
    """
    if a.shape != b.shape:
        raise ValueError(f"Batch mean dice coefficient expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")

    dices = []
    for a, b, in zip(a, b):
        dices.append(dice(a, b))
    mean_dice = np.mean(dices)
    return mean_dice

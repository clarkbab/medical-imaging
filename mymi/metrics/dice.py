import numpy as np
import SimpleITK as sitk
import torch

def dice(
    a: np.ndarray,
    b: np.ndarray) -> float:
    """
    returns: the dice score.
    args:
        a: an array.
        b: another array.
    """
    if a.shape != b.shape:
        raise ValueError(f"Got mismatched input shapes: '{a.shape}' and '{b.shape}'.")

    # Get cardinality.
    a_card = a.sum()
    b_card = b.sum()

    # Get number of intersecting pixels.
    int_card = (a * b).sum()

    # Calculate dice score.
    dice = 2. * int_card / (a_card + b_card)

    return dice

def batch_mean_dice(
    a: np.ndarray,
    b: np.ndarray) -> float:
    """
    returns: returns the mean batch DSC.
    args:
        a: a 4D array (batch of 3D binary volumes).
        b: a 4D array (batch of 3D binary volumes).
    """
    assert a.shape == b.shape

    # Get cardinality.
    a_card = a.sum((1, 2, 3))
    b_card = b.sum((1, 2, 3))

    # Get number of intersecting pixels.
    int_card = (a * b).sum((1, 2, 3)) 

    # Calculate dice score.
    dice_scores = 2. * int_card / (a_card + b_card)

    return dice_scores.mean()

def sitk_batch_mean_dice(
        a: torch.Tensor,
        b: torch.Tensor) -> torch.Tensor:
    dice_scores = []
    for i in range(len(a)):
        img_a_i = sitk.GetImageFromArray(a[i])
        img_b_i = sitk.GetImageFromArray(b[i])
        ol_filter = sitk.LabelOverlapMeasuresImageFilter()
        ol_filter.Execute(img_a_i, img_b_i)
        dice = ol_filter.GetDiceCoefficient()
        dice_scores.append(dice)
    return torch.Tensor(np.mean(dice_scores))

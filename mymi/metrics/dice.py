import numpy as np
import SimpleITK as sitk
import torch

def batch_dice(a, b):
    """
    returns: the mean dice similarity coefficient (DSC) for the batch.
    args:
        a: a batch of 3D binary volumes.
        b: a bath of 3D binary volumes.
    """
    assert a.shape == b.shape

    # Get cardinality.
    a_card = a.sum((1, 2, 3))
    b_card = b.sum((1, 2, 3))

    # Get number of intersecting pixels.
    int_card = (a * b).sum((1, 2, 3)) 

    # Calculate dice score.
    dice = 2. * int_card / (a_card + b_card)

    # Handle nan cases.
    # For each prediction in batch, handle case where dice is 'nan'. This
    # occurs when no foreground label or predictions are made, e.g. for
    # unlabelled volumes.
    for i, d in enumerate(dice):
        if np.isnan(d):
            dice[i] = 1

    # Average dice score across batch.
    mean_dice = dice.mean()

    return mean_dice

def sitk_batch_dice(a, b):
    dices = []
    for i in range(len(a)):
        img_a_i = sitk.GetImageFromArray(a[i])
        img_b_i = sitk.GetImageFromArray(b[i])
        ol_filter = sitk.LabelOverlapMeasuresImageFilter()
        ol_filter.Execute(img_a_i, img_b_i)
        dice = ol_filter.GetDiceCoefficient()
        dices.append(dice)
    return np.mean(dices)

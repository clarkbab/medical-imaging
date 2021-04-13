import torch

def batch_dice(pred, label):
    """
    returns: the mean dice similarity coefficient (DSC) for the batch.
    args:
        pred: a batch of network predictions, e.g. shape (n, 2, 512, 512, 212).
        label: a batch of labels, e.g. shape (n, 512, 512, 212).
    """
    # If pred hasn't been binarised, then do so.
    if len(pred.shape) == 5:
        pred = pred.argmax(dim=1)

    assert pred.shape == label.shape

    # Get pred, label cardinality.
    pred_card = pred.sum(dim=(1, 2, 3))
    label_card = label.sum(dim=(1, 2, 3))

    # Get number of intersecting pixels.
    int_card = (pred * label).sum(dim=(1, 2, 3)) 

    # Calculate dice score.
    dice = 2. * int_card / (pred_card + label_card)

    # Handle nan cases.
    # For each prediction in batch, handle case where dice is 'nan'. This
    # occurs when no foreground label or predictions are made, e.g. for
    # unlabelled volumes.
    for i, d in enumerate(dice):
        if torch.isnan(d):
            dice[i] = 1

    # Average dice score across batch.
    mean_dice = dice.mean()

    return mean_dice

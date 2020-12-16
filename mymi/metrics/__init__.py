import torch

def dice(pred, label):
    """
    pred: a batch of network predictions, e.g. shape (n, 2, 512, 512).
    label: a batch of labels, e.g. shape (n, 512, 512).
    """
    # Get binary predictions for network.
    pred_bin = pred.argmax(dim=1)

    assert pred_bin.shape == label.shape

    # Get pred, label cardinality.
    pred_card = pred_bin.sum(dim=(1, 2))
    label_card = label.sum(dim=(1, 2))

    # Get number of intersecting pixels.
    int_card = (pred_bin * label).sum(dim=(1, 2)) 

    # Calculate dice score.
    dice = 2. * int_card / (pred_card + label_card)

    # Handle nan cases.
    for i, d in enumerate(dice):
        if torch.isnan(d):
            dice[i] = 1

    return dice.mean()


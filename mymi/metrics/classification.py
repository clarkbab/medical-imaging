import numpy as np

def true_negative_rate(
    pred: np.ndarray,
    label: np.ndarray) -> float:
    if pred.shape != label.shape:
        raise ValueError(f"Metric 'true_negative_rate' expects arrays of equal shape. Got '{pred.shape}' and '{label.shape}'.")
    background = (1 - label)
    true_neg = (1 - pred) * background
    tnr = true_neg.sum() / background.sum()
    return tnr
    
def true_positive_rate(
    pred: np.ndarray,
    label: np.ndarray) -> float:
    if pred.shape != label.shape:
        raise ValueError(f"Metric 'true_positive_rate' expects arrays of equal shape. Got '{pred.shape}' and '{label.shape}'.")
    true_pos = pred * label
    # Doesn't make sense to average over all voxels, as TP would be small even if perfect prediction.
    tpr = true_pos.sum() / label.sum()
    return tpr

def false_negative_rate(
    pred: np.ndarray,
    label: np.ndarray) -> float:
    if pred.shape != label.shape:
        raise ValueError(f"Metric 'false_negative_rate' expects arrays of equal shape. Got '{pred.shape}' and '{label.shape}'.")
    false_neg = (1 - pred) * label
    fnr = false_neg.sum() / label.sum()
    return fnr

def false_positive_rate(
    pred: np.ndarray,
    label: np.ndarray) -> float:
    if pred.shape != label.shape:
        raise ValueError(f"Metric 'false_positive_rate' expects arrays of equal shape. Got '{pred.shape}' and '{label.shape}'.")
    background = (1 - label)
    false_pos = pred * background
    fpr = false_pos.sum() / background.sum()
    return fpr

import numpy as np

def true_negatives(
    a: np.ndarray,
    b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'true_negatives' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    return np.sum((1 - a) * (1 - b)) / a.size

def true_positives(
    a: np.ndarray,
    b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'true_positives' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    return np.sum(a * b) / a.size

def false_negatives(
    a: np.ndarray,
    b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'false_negatives' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    return np.sum((1 - a) * b) / a.size

def false_positives(
    a: np.ndarray,
    b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'false_positives' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    return np.sum(a * (1 - b)) / a.size

import numpy as np

from mymi.typing import ImageSpacing3D

def ncc(
    a: np.ndarray,
    b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'ncc' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")
    if (a.dtype != np.float32 and a.dtype != np.float64) or (b.dtype != np.float32 and b.dtype != np.float64):
        raise ValueError(f"Metric 'ncc' expects float32/64 arrays. Got '{a.dtype}' and '{b.dtype}'.")

    # Calculate normalised cross-correlation.
    norm_a = (a - np.mean(a)) / np.std(a)
    norm_b = (b - np.mean(b)) / np.std(b)
    result = (1 / a.size) * np.sum(norm_a * norm_b)

    return result

def tre(
    a: np.ndarray,
    b: np.ndarray,
    spacing: ImageSpacing3D) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Metric 'tre' expects arrays of equal shape. Got '{a.shape}' and '{b.shape}'.")

    # Calculate euclidean distances.
    tres = []
    for ai, bi in zip(a, b):
        tre = np.linalg.norm((bi - ai) * spacing)
        tres.append(tre)

    stats = {
        'tre-mean': np.mean(tres),
        'tre-min': np.min(tres),
        'tre-max': np.max(tres),
        'tre-median': np.median(tres),
        'tre-95': np.quantile(tres, 0.95)
    }
    
    return stats

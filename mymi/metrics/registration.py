import numpy as np

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

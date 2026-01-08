import numpy as np
from typing import *

from mymi.typing import *
from mymi.utils import *

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
    a: Union[List[Point], PointsArray, LandmarksFrame],
    b: Union[List[Point], PointsArray, LandmarksFrame],
    ) -> Union[np.ndarray, pd.DataFrame]:
    assert len(a) == len(b)

    if isinstance(a, pd.DataFrame):
        assert isinstance(b, pd.DataFrame)
        return_type = 'frame'
    else:
        return_type = 'array'

    if return_type == 'frame':
        tre_df = a.merge(b, on=['patient-id', 'landmark-id'])
        assert len(tre_df) == len(a), "Must have been different landmarks in a and b."
        for i in range(3):
            tre_df[f'diff-{i}'] = np.abs(tre_df[f'{i}_x'] - tre_df[f'{i}_y'])
        tre_df['tre'] = np.sqrt(tre_df['diff-0'] ** 2 + tre_df['diff-1'] ** 2 + tre_df['diff-2'] ** 2)
        tre_df = tre_df[['patient-id', 'landmark-id', 'diff-0', 'diff-1', 'diff-2', 'tre']]
        return tre_df
    else:
        a, b = np.array(a), np.array(b)
        return np.sqrt(((b - a) ** 2).sum(axis=1))

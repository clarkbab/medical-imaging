from collections.abc import Iterable, Sequence as CSequence
from GPUtil import getGPUs
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
# from pynvml import nvidia_smi
import sys
from time import perf_counter
from typing import *

from mymi import config
from mymi import logging
from mymi.typing import *

def append_dataframe(df: pd.DataFrame, odf: pd.DataFrame) -> pd.DataFrame:
    # Pandas doesn't preserve index name when names are different between concatenated dataframes,
    # this could occur when one of the dataframes is empty.
    index_name = None
    if len(df) == 0 and odf.index.name is not None:
        index_name = odf.index.name
    elif len(odf) == 0 and df.index.name is not None:
        index_name = df.index.name

    # Perform concatenation.
    df = pd.concat((df, odf), axis=0)
    
    # Update index name.
    if index_name is not None:
        df.index.name = index_name

    return df

def deep_merge(d: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    all_keys = list(set(d.keys()).union(set(default.keys())))
    merged = {}
    for k in all_keys:
        if k in d and k in default:
            if isinstance(default[k], dict):
                assert isinstance(d[k], dict)
                merged[k] = deep_merge(d[k], default[k])
            elif isinstance(default[k], (bool, int, float, str)):
                merged[k] = d[k] if k in d else default[k]
            else:
                raise ValueError(f"Unrecognised type for default key '{k}'.")
        elif k in d:
            merged[k] = d[k]
        else:
            merged[k] = default[k]
            
    return merged

def encode(o: Any) -> str:
    return hashlib.sha1(json.dumps(o).encode('utf-8')).hexdigest()

def escape_filepath(f: str) -> str:
    if f.split(':')[0] == 'files':
        f = f.split(':')[1]
        f = os.path.join(config.directories.files, f)
    return f

def get_axis_name(
    view: int,
    abbreviate: bool = False) -> str:
    if view == 0:
        return 'sag.' if abbreviate else 'sagittal'
    elif view == 1:
        return 'cor.' if abbreviate else 'coronal'
    elif view == 2:
        return 'ax.' if abbreviate else 'axial'

def handle_idx_prefix(
    id: str,
    list_ids: Callable) -> id:
    if id.startswith('idx:'):
        idx = int(id.split(':')[1])
        ids = list_ids() 
        if idx > len(ids) - 1:
            print(ids)
            raise ValueError(f"Index ({idx}) was larger than list (len={len(ids)}).")
        id = ids[idx]

    return id

def is_generic(t: Any) -> bool:
    return get_origin(t) is not None

def is_windows() -> bool:
    return 'win' in sys.platform

def isinstance_generic(a: Any, t: Any) -> bool:
    # Checks if 'a' is of type 't' for generic (e.g. List[], Dict[]) and
    # non-generic types.
    if t is None:
        return a is None
    if not is_generic(t):
        return isinstance(a, t)
    
    # Check main type - e.g. 'list' for List[str], or 'union' for Union[str, int].
    main_type = get_origin(t)

    if main_type is Literal:
        # Check for literal matches.
        literals = get_args(t)
        for l in literals:
            if a == l:
                return True
        return False
    
    if main_type is Union:
        # Check for any matching subtype.
        subtypes = get_args(t)
        for s in subtypes:
            if isinstance_generic(a, s):
                return True
        return False
    
    # If not a Union main type, then main type must match.
    if not isinstance(a, main_type):
        return False
    
    if main_type in (list, CSequence):
        # For iterable main types (one subtype only - e.g. List[str] or List[int]),
        # check that all elements in 'a' match the required subtype.
        subtype = get_args(t)[0]
        for ai in a:
            if not isinstance_generic(ai, subtype):
                return False
    elif main_type in (dict,):
        # For dict main types (key/value subtypes - e.g. Dict[str, int]),
        # check that all keys/values in 'a' match the required subtypes.
        k_subtype, v_subtype = get_args(t)
        for k, v in a.items():
            if not isinstance_generic(k, k_subtype) or not isinstance(v, v_subtype):
                return False
            
    return True

def get_batch_centroids(label_batch, plane):
    """
    returns: the centroid location of the label along the plane axis, for each
        image in the batch.
    args:
        label_batch: the batch of labels.
        plane: the plane along which to find the centroid.
    """
    assert plane in ('axial', 'coronal', 'sagittal')

    # Move data to CPU.
    label_batch = label_batch.cpu()

    # Determine axes to sum over.
    if plane == 'axial':
        axes = (0, 1)
    elif plane == 'coronal':
        axes = (0, 2)
    elif plane == 'sagittal':
        axes = (1, 2)

    centroids = np.array([], dtype=np.int)

    # Loop through batch and get centroid for each label.
    for label_i in label_batch:
        # Get weighting along 'plane' axis.
        weights = label_i.sum(axes)

        # Get average weighted sum.
        indices = np.arange(len(weights))
        avg_weighted_sum = (weights * indices).sum() /  weights.sum()

        # Get centroid index.
        centroid = np.round(avg_weighted_sum).long()
        centroids = np.append(centroids, centroid)

    return centroids

def fplot(
    f_str: str, 
    figsize: Tuple[float, float] = (8, 6),
    x: Optional[List[float]] = None,
    y: Optional[List[float]] = None, 
    xres: float = 1e-1,
    xlim: Tuple[float, float] = (-10, 10),
    **kwargs) -> None:
    # Rename x so it can be used in 'eval'.
    x_data, y_data = x, y
    
    # Replace params in 'f'.
    f = f_str
    params = dict(((k, v) for k, v in kwargs.items() if len(k) == 1 and k not in ('x', 'y')))
    for k, v in params.items():
        f = f.replace(k, str(v))

    # Plot function.
    x = np.linspace(xlim[0], xlim[1], int((xlim[1] - xlim[0]) / xres))
    y = eval(f)
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    
    # Plot points.
    if x_data is not None or y_data is not None:
        assert x_data is not None and y_data is not None
        assert len(x_data) == len(y_data)
        plt.scatter(x_data, y_data, marker='x')
        
    param_str = ','.join((f'{k}={v:.3f}' for k, v in params.items()))
    plt.title(f"{f_str}, {param_str}")

    plt.show()

def gpu_count() -> int:
    return len(getGPUs())

def gpu_usage() -> List[float]:
    return [g.memoryUsed for g in getGPUs()]

def gpu_usage_nvml() -> List[float]:
    usages = []
    # nvsmi = nvidia_smi.getInstance()
    results = nvsmi.DeviceQuery('memory.used')['gpu']
    for result in results:
        assert result['fb_memory_usage']['unit'] == 'MiB'
        usage = result['fb_memory_usage']['used']
        usages.append(usage)
    return usages

def benchmark(
    f: Callable,
    args: Tuple = (),
    after: Optional[Callable] = None,
    before: Optional[Callable] = None,
    n: int = 100) -> float:
    if before is not None:
        before()

    # Evaluate function 'n' times.
    durations = [] 
    for _ in range(n):
        start = perf_counter()
        f(*args)
        durations.append(perf_counter() - start)

    if after is not None:
        after()

    return np.mean(durations)

def p_landmarks(landmarks: List[LandmarkID], f: float) -> List[LandmarkID]:
    # Take non-random subset of landmarks.
    n_landmarks = int(f * len(landmarks))
    idxs = np.linspace(0, len(landmarks), n_landmarks).astype(int)
    landmarks = [l for i, l in enumerate(landmarks) if i in idxs]
    return landmarks

def reverse_xy(data: Union[Sequence, np.ndarray]) -> Union[Sequence, np.ndarray]:
    if isinstance(data, np.ndarray):
        if data.shape == (3, 3):
            data[0][0], data[1][1] = -data[0][0], -data[1][1]
        elif data.shape == (9,):
            data[0], data[4] = -data[0], -data[4]
        elif data.shape == (3,):
            data[0], data[1] = -data[0], -data[1]
        else:
            raise ValueError(f"Expected data to be 3 or 9 elements, got {data.shape}.")
    elif isinstance(data, tuple):
        data = list(data)
        if len(data) == 3:
            data[0], data[1] = -data[0], -data[1]
        elif len(data) == 9:
            data[0], data[4] = -data[0], -data[4]
        else:
            raise ValueError(f"Expected data to be 3 or 9 elements, got {len(data)}.")
        data = tuple(data)
    else:
        raise ValueError(f"Expected data to be a numpy array or tuple, got {type(data)}.")
    return data

def transpose_image(
    data: Union[ImageArray, VectorImageArray],
    vector: bool = False) -> Union[ImageArray, VectorImageArray]:
    # Transposes spatial coordinates, whilst maintaining vector dimension as first dim.
    if vector and data.shape[0] != 3:
        raise ValueError(f"Expected vector dimension first, got {data.shape}.")
    data = np.transpose(data)
    if vector:
        data = np.moveaxis(data, -1, 0)
    return data

def with_dry_run(
    dry_run: bool,
    f: Callable,
    msg: str = None) -> None:
    if dry_run:
        if msg is not None:
            logging.info(f"Dry run: {msg}")
    else:
        f()
        if msg is not None:
            logging.info(msg)

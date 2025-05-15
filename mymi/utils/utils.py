from collections.abc import Iterable, Sequence as CSequence
from GPUtil import getGPUs
import hashlib
import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pynvml.smi import nvidia_smi
from time import perf_counter, time
from typing import *

from mymi import config
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

def is_generic(t: Any) -> bool:
    return get_origin(t) is not None

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

def arg_assert_lengths(args: List[List[Any]]) -> None:
    arg_0 = args[0]
    for arg in args[1:]:
        if len(arg) != len(arg_0):
            raise ValueError(f"Expected arg lengths to match. Length of arg '{arg}' didn't match '{arg_0}'.")

def arg_assert_literal(
    arg: Any,
    literal: Union[Any, List[Any]]) -> None:
    literals = arg_to_list(literal, type(arg))
    if arg not in literals:
        raise ValueError(f"Expected argument to be one of '{literals}', got '{arg}'.")

def arg_assert_literal_list(
    arg: Union[Any, List[Any]],
    arg_type: Any,
    literal: Union[Any, List[Any]]) -> None:
    args = arg_to_list(arg, arg_type)
    literals = arg_to_list(literal, arg_type)
    for arg in args:
        if arg not in literals:
            raise ValueError(f"Expected argument to be one of '{literals}', got '{arg}'.")

def arg_assert_present(
    arg: Any,
    name: str) -> None:
    if arg is None:
        raise ValueError(f"Argument '{name}' expected not to be None.")

def arg_to_list(
    arg: Optional[Any],
    arg_type: Union[Any, Tuple[Any]],
    length: int = 1,      # Expand a matching type to multiple elements, e.g. None -> [None, None, None].
    literals: Dict[str, Tuple[Any]] = {},
    out_type: Optional[Any] = None) -> List[Any]:
    # Allow multiple types in 'arg_type'.
    # E.g. patient ID can be str/int, colours can be str/tuple.
    if type(arg_type) is tuple:
        # if out_type is None:
        #     raise ValueError(f"Must specify 'out_type' when multiple input types are used ({arg_type}).")
        arg_types = arg_type
    else:
        arg_types = (arg_type,)
    
    # Check literal matches.
    literal_types = (int, str) 
    if type(arg) in literal_types and arg in literals:
        arg = literals[arg]
        # If arg is a function, run it now. This means the function
        # is not evaluated every time 'arg_to_list' is called, only when
        # the arg matches the appropriate literal (e.g. 'all').
        if callable(arg):
            arg = arg()

        return arg

    # Check types.
    matched = False
    for a in arg_types:
        if isinstance_generic(arg, a):
            matched = True
            arg = [arg] * length
            break
        
    # Convert to output type.
    if matched and out_type is not None:
        arg = [out_type(a) for a in arg]

    return arg

def arg_broadcast(
    arg: Any,
    b_arg: Any,
    arg_type: Optional[Any] = None,
    out_type: Optional[Any] = None):
    # Convert arg to list.
    if arg_type is not None:
        arg = arg_to_list(arg, arg_type, out_type=out_type)

    # Get broadcast length.
    b_len = b_arg if type(b_arg) is int else len(b_arg)

    # Broadcast arg.
    if isinstance(arg, Iterable) and not isinstance(arg, str) and len(arg) == 1 and b_len != 1:
        arg = b_len * arg
    elif not isinstance(arg, Iterable) or (isinstance(arg, Iterable) and isinstance(arg, str)):
        arg = b_len * [arg]

    return arg

def gpu_count() -> int:
    return len(getGPUs())

def gpu_usage() -> List[float]:
    return [g.memoryUsed for g in getGPUs()]

def gpu_usage_nvml() -> List[float]:
    usages = []
    nvsmi = nvidia_smi.getInstance()
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

def p_landmarks(landmarks: List[Landmark], f: float) -> List[Landmark]:
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
    data: np.ndarray,
    vector: bool = False) -> np.ndarray:
    # Transposes spatial coordinates, whilst maintaining vector dimension as first dim.
    data = np.transpose(data)
    if vector:
        assert data.shape[-1] == 3
        data = np.moveaxis(data, -1, 0)
    return data

def get_view_name(
    view: int,
    abbreviate: bool = True) -> str:
    if view == 0:
        return 'sag.' if abbreviate else 'sagittal'
    elif view == 1:
        return 'cor.' if abbreviate else 'coronal'
    elif view == 2:
        return 'ax.' if abbreviate else 'axial'

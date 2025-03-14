from collections.abc import Iterable, Sequence as CSequence
from contextlib import contextmanager
from GPUtil import getGPUs
import hashlib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pynvml.smi import nvidia_smi
from time import perf_counter, time
from typing import *

# Commented due to circular import.
# from mymi.loaders import Loader
# from mymi import logging
from mymi import config

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

def encode(o: Any) -> str:
    return hashlib.sha1(json.dumps(o).encode('utf-8')).hexdigest()

# Commented due to circular import.
# def get_manifest():
#     datasets = ['PMCC-HN-TEST-LOC', 'PMCC-HN-TRAIN-LOC']
#     region = 'BrainStem'
#     n_folds = 5
#     n_train = 5
#     test_fold = 0
#     _, _, test_loader = Loader.build_loaders(datasets, region, load_test_origin=False, n_folds=n_folds, n_train=n_train, test_fold=test_fold)
#     samples = []
#     for ds_b, samp_b in iter(test_loader):
#         samples.append((ds_b[0], samp_b[0].item()))
#     return samples

def is_generic(t: Any) -> bool:
    return get_origin(t) is not None

def isinstance_generic(a: Any, t: Any) -> bool:
    # Checks if 'a' is of type 't' for generic (e.g. List[], Dict[]) and
    # non-generic types.
    if t is None:
        return a is None
    if not is_generic(t):
        return isinstance(a, t)
    
    # Check parent type.
    origin = get_origin(t)
    
    # Union - Check if any subtype matches.
    if origin is Union:
        subtypes = get_args(t)
        for s in subtypes:
            if isinstance_generic(a, s):
                return True
        return False
    
    # For non-Union types, check parent type matches.
    if not isinstance(a, origin):
        return False
    
    # Iterable - Check if all elements match type.
    if origin in (list, CSequence):
        # Check all elements.
        subtype = get_args(t)[0]
        for ai in a:
            if not isinstance_generic(ai, subtype):
                return False
    elif origin in (dict,):
        # Check all keys/values.
        ktype, vtype = get_args(t)
        for k, v in a.items():
            if not isinstance_generic(k, ktype) or not isinstance(v, vtype):
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

def save_csv(
    data: pd.DataFrame,
    *path: List[str],
    index: bool = False,
    header: bool = True,
    overwrite: bool = True) -> None:
    filepath = os.path.join(config.directories.files, *path)
    dirpath = os.path.dirname(filepath)
    if os.path.exists(filepath):
        if overwrite:
            os.makedirs(dirpath, exist_ok=True)
            data.to_csv(filepath, header=header, index=index)
        else:
            raise ValueError(f"File '{filepath}' already exists, use overwrite=True.")
    else:
        os.makedirs(dirpath, exist_ok=True)
        data.to_csv(filepath, header=header, index=index)

def load_csv(
    *path: List[str],
    exists_only: bool = False,
    map_cols: Dict[str, str] = {},
    map_landmark_id: bool = True,
    map_patient_id: bool = True,
    map_types: Dict[str, Any] = {},
    **kwargs: Dict[str, str]) -> Optional[pd.DataFrame]:
    filepath = os.path.join(config.directories.files, *path)
    if not os.path.exists(filepath):
        if exists_only:
            return False
        else:
            raise ValueError(f"CSV at filepath '{filepath}' not found.")
    elif exists_only:
        return True

    # Load CSV.
    if map_landmark_id:
        map_types['landmark-id'] = str
    if map_patient_id:
        map_types['patient-id'] = str

    df = pd.read_csv(filepath, dtype=map_types, **kwargs)

    # Map column names.
    df = df.rename(columns=map_cols)

    return df

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
    out_type: Optional[Any] = None,) -> List[Any]:
    # Allow multiple types in 'arg_type'.
    # E.g. patient ID can be str/int, colours can be str/tuple.
    if type(arg_type) is tuple:
        # if out_type is None:
        #     raise ValueError(f"Must specify 'out_type' when multiple input types are used ({arg_type}).")
        arg_types = arg_type
    else:
        arg_types = (arg_type,)

    # Check types.
    matched = False
    for a in arg_types:
        if isinstance_generic(arg, a):
            matched = True

            literal_types = (int, str) 
            if type(arg) in literal_types and arg in literals:
                arg = literals[arg]
                # If arg is a function, run it now. This means the function
                # is not evaluated every time 'arg_to_list' is called, only when
                # the arg matches the appropriate literal (e.g. 'all').
                if callable(arg):
                    arg = arg()
            else:
               arg = [arg] * length
        
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

# Time for each 'recorded' event is stored in a row of the CSV.
# Additional columns can be populated using 'data'.
class Timer:
    def __init__(
        self,
        columns: Dict[str, str] = {}):
        self.__cols = columns
        self.__cols['time'] = float
        self.__df = pd.DataFrame(columns=self.__cols.keys())

    @contextmanager
    def record(
        self,
        data: Dict[str, Union[str, int, float]] = {},
        enabled: bool = True):
        try:
            if enabled:
                start = time()
            yield None
        finally:
            if enabled:
                print(data)
                data['time'] = time() - start
                self.__df = append_row(self.__df, data)

    def save(self, filepath):
        self.__df.astype(self.__cols).to_csv(filepath, index=False)

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

def transpose_image(
    data: np.ndarray,
    is_vector: bool = False) -> np.ndarray:
    # Transposes spatial coordinates, whilst maintaining vector dimension as last dim.
    data = np.transpose(data)
    if is_vector:
        data = np.moveaxis(data, 0, -1)
    return data

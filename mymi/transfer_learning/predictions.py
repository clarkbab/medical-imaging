import numpy as np
import os
import pandas as pd
from scipy.optimize import least_squares
from tqdm import tqdm
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from mymi import config
config.Regions.mode = 1
from mymi.evaluation.dataset.nifti import load_segmenter_evaluation
from mymi.loaders import Loader, get_n_train_max
from mymi import logging
from mymi.metrics import higher_is_better
from mymi.regions import RegionNames
from mymi.regions.tolerances import get_region_tolerance
from mymi.utils import arg_assert_literal, arg_assert_literal_list, arg_broadcast, arg_to_list, encode

DEFAULT_FONT_SIZE = 8
DEFAULT_MAX_NFEV = int(1e6)
DEFAULT_METRIC_LEGEND_LOCS = {
    'dice': 'lower right',
    'hd': 'upper right',
    'hd-95': 'upper right',
    'msd': 'upper right'
}
DEFAULT_METRIC_LABELS = {
    'dice': 'DSC',
    'hd': 'HD (mm)',
    'hd-95': '95HD (mm)',
    'msd': 'MSD (mm)',
}
DEFAULT_METRIC_Y_LIMS = {
    'dice': (0, 1),
    'hd': (0, None),
    'hd-95': (0, None),
    'msd': (0, None)
}
for region in RegionNames:
    tol = get_region_tolerance(region)
    DEFAULT_METRIC_LEGEND_LOCS[f'apl-mm-tol-{tol}'] = 'upper right'
    DEFAULT_METRIC_LEGEND_LOCS[f'dm-surface-dice-tol-{tol}'] = 'lower right'
    DEFAULT_METRIC_LABELS[f'apl-mm-tol-{tol}'] = fr'APL, $\tau$={tol}mm (mm)'
    DEFAULT_METRIC_LABELS[f'dm-surface-dice-tol-{tol}'] = fr'Surface DSC, $\tau$={tol}mm'
    DEFAULT_METRIC_Y_LIMS[f'apl-mm-tol-{tol}'] = (0, None)
    DEFAULT_METRIC_Y_LIMS[f'dm-surface-dice-tol-{tol}'] = (0, 1)

DEFAULT_N_SAMPLES = int(1e4)
DEFAULT_N_TRAINS = [5, 10, 20, 50, 100, 200, 400, 800, None]
LOG_SCALE_X_UPPER_LIMS = [100, 150, 200, 300, 400, 600, 800]
LOG_SCALE_X_TICK_LABELS = [5, 10, 20, 50, 100, 200, 400, 800]

def load_evaluation_data(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model_type: Union[Literal['clinical', 'transfer'], List[Literal['clinical', 'transfer']]],
    metric: str,
    n_train: Union[Optional[int], List[Optional[int]]] = DEFAULT_N_TRAINS,
    n_folds: Optional[int] = 5,
    test_fold: Union[int, List[int]] = list(range(5))) -> pd.DataFrame:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    model_types = arg_to_list(model_type, str)
    n_trains = arg_to_list(n_train, int)
    test_folds = arg_to_list(test_fold, int)

    # Load evaluations and combine.
    dfs = []
    for region in regions:
        localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')

        for test_fold in test_folds:
            # Add public evaluation as 'transfer' with 'n=0'.
            if 'transfer' in model_types:
                seg_run = 'public-1gpu-150epochs'
                segmenter = (f'segmenter-{region}', seg_run, 'BEST')
                df = load_segmenter_evaluation(datasets, localiser, segmenter, n_folds=n_folds, test_fold=test_fold)
                df = df[df['metric'] == metric]
                df['model-type'] = 'transfer'
                df['n-train'] = 0
                dfs.append(df)

            # Get number of training cases.
            n_train_max = get_n_train_max(datasets, region, n_folds=n_folds, test_fold=test_fold)

            # Add clinical/transfer evaluations.
            for model_type in (model_type for model_type in model_types if model_type != 'public'):
                for n_train in n_trains:
                    # Skip if we've exceeded available number of training samples.
                    if n_train is not None and n_train >= n_train_max:
                        continue

                    seg_run = f'{model_type}-fold-{test_fold}-samples-{n_train}'
                    segmenter = (f'segmenter-{region}-v2', seg_run, 'BEST')
                    df = load_segmenter_evaluation(datasets, localiser, segmenter, test_fold=test_fold)
                    df = df[df['metric'] == metric]
                    df['model-type'] = model_type
                    df['n-train'] = n_train
                    dfs.append(df)
                   
    # Save dataframe.
    df = pd.concat(dfs, axis=0)

    # Replace `n_train=None` with true value.
    none_nums = {}
    for region in regions:
        tl, vl, _ = Loader.build_loaders(datasets, region, n_folds=n_folds, test_fold=0)
        n_train = len(tl) + len(vl)
        none_nums[region] = n_train
    df.loc[df['n-train'].isnull(), 'n-train'] = df[df['n-train'].isnull()].region.apply(lambda r: none_nums[r])
    df['n-train'] = df['n-train'].astype(int)

    # Add model names.
    df['model'] = df['model-type'] + '-' + df['n-train'].astype(str)

    # Sort values.
    df = df.sort_values(['fold', 'region', 'model-type', 'n-train'])

    return df

def load_raw_evaluation_data(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model_type: Union[Literal['clinical', 'transfer'], List[Literal['clinical', 'transfer']]],
    metric: str) -> np.ndarray:
    n_folds = 5
    df = load_evaluation_data(dataset, region, model_type, metric)
    n_trains = df['n-train'].unique()
    df = df.sort_values(['n-train', 'fold'])
    shape = (len(n_trains), n_folds, -1)
    raw_data = df['value'].to_numpy().reshape(shape)
    return raw_data, n_trains

def create_bootstrap_samples(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model_type: Union[Literal['clinical', 'transfer'], List[Literal['clinical', 'transfer']]],
    metric: Union[str, List[str]],
    stat: Union[str, List[str]],
    n_samples: int = DEFAULT_N_SAMPLES) -> None:
    model_types = arg_to_list(model_type, str)
    metrics = arg_to_list(metric, str)
    stats = arg_to_list(stat, str)
    logging.arg_log('Creating bootstrap predictions', ('dataset', 'region', 'model_type', 'metric', 'stat'), (dataset, region, model_type, metric, stat))

    for model_type in model_types:
        for metric in metrics:
            for stat in stats:
                if '{tol}' in metric:
                    metric = metric.replace('{tol}', str(get_region_tolerance(region)))

                # Load raw data.
                raw_data, n_trains = load_raw_evaluation_data(dataset, region, model_type, metric)

                # Create bootstrap samples.
                shape = raw_data.shape
                boot_samples = []
                for n_train in tqdm(range(shape[0])):
                    boot_sample_folds = []
                    for fold in range(shape[1]):
                        boot_sample_fold = np.random.choice(raw_data[n_train, fold], size=(shape[2], n_samples), replace=True)
                        boot_sample_folds.append(boot_sample_fold)
                    boot_sample = np.stack(boot_sample_folds, axis=0)
                    boot_samples.append(boot_sample)
                boot_samples = np.stack(boot_samples, axis=0)

                if stat == 'mean':
                    boot_samples = boot_samples.mean(axis=2)
                elif stat == 'q1':
                    boot_samples = np.quantile(boot_samples, 0.25, axis=2)
                elif stat == 'q3':
                    boot_samples = np.quantile(boot_samples, 0.75, axis=2)

                filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'samples', encode(dataset), region, model_type, metric, stat, 'samples.npz')
                if not os.path.exists(os.path.dirname(filepath)):
                    os.makedirs(os.path.dirname(filepath))
                np.savez_compressed(filepath, data=boot_samples, n_trains=n_trains)

def load_bootstrap_samples(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model_type: Union[Literal['clinical', 'transfer'], List[Literal['clinical', 'transfer']]],
    metric: str,
    stat: str,
    n_samples: int = DEFAULT_N_SAMPLES) -> np.ndarray:
    filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'samples', encode(dataset), region, model_type, metric, stat, 'samples.npz')
    data = np.load(filepath)
    return data['data'], data['n_trains']

def create_bootstrap_predictions(
    dataset: Union[str, List[str]],
    region: str,
    model_type: Union[str, List[str]],
    metric: Union[str, List[str]], 
    stat: Union[str, List[str]],
    n_samples: int = DEFAULT_N_SAMPLES,
    raise_error: bool = True) -> None:
    model_types = arg_to_list(model_type, str)
    metrics = arg_to_list(metric, str)
    stats = arg_to_list(stat, str)
    logging.arg_log('Creating bootstrap predictions', ('dataset', 'region', 'model_type', 'metric', 'stat'), (dataset, region, model_type, metric, stat))

    for model_type in model_types:
        for metric in metrics:
            for stat in stats:
                if '{tol}' in metric:
                    metric = metric.replace('{tol}', str(get_region_tolerance(region)))

                # Load samples.
                samples, n_trains = load_bootstrap_samples(dataset, region, model_type, metric, stat, n_samples=n_samples)

                # Get placeholders.
                n_preds = np.max(n_trains) + 1
                n_samples = samples.shape[-1]
                n_folds = samples.shape[1]
                n_params = 3
                params = np.zeros((n_samples, n_params))
                preds = np.zeros((n_samples, n_preds))

                for i in tqdm(range(n_samples)):
                    # Flatten data.
                    x = np.array([])
                    y = np.array([])
                    for j, n_train in enumerate(n_trains):
                        x = np.concatenate((x, n_train * np.ones(n_folds)))
                        y = np.concatenate((y, samples[j, :, i]))
                        
                    # Fit curve.
                    try:
                        p_init = __get_p_init(metric)
                        p_opt, _, _ = __fit_curve(p_init, x, y)
                    except ValueError as e:
                        if raise_error:
                            logging.error(f"Error when fitting sample '{i}':")
                            raise e
                        else:
                            return x, y, w
                    params[i] = p_opt
                    
                    # Create prediction points.
                    x = np.linspace(0, n_preds - 1, num=n_preds)
                    y = __f(x, p_opt)
                    preds[i] = y

                # Set 'clinical' model predictions to 'NaN' as 'clinical' model must be trained on
                # 5 or more training cases.
                if model_type == 'clinical':
                    preds[:, :5] = np.nan
                    
                # Save data.
                filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'preds', encode(dataset), region, model_type, metric, stat, 'preds.npz')
                if not os.path.exists(os.path.dirname(filepath)):
                    os.makedirs(os.path.dirname(filepath))
                np.savez_compressed(filepath, data=preds, params=params)

def load_bootstrap_predictions(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model_type: Union[Literal['clinical', 'transfer'], List[Literal['clinical', 'transfer']]],
    metric: str,
    stat: str,
    n_samples: int = DEFAULT_N_SAMPLES) -> Tuple[np.ndarray, np.ndarray]:
    filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'preds', encode(dataset), region, model_type, metric, stat, 'preds.npz')
    data = np.load(filepath)
    preds = data['data']
    params = data['params']
    return preds, params

def load_bootstrap_point_predictions(
    dataset: Union[str, List[str]],
    region: str,
    model_type: str,
    n: Union[int, Literal['all']],
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    include_params: bool = False,
    n_samples: int = DEFAULT_N_SAMPLES) -> Tuple[np.ndarray]:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))
    data, params = load_bootstrap_predictions(dataset, region, model_type, metric, stat, include_params=include_params, n_samples=n_samples)
    if n == 'all':
        n = -1
    data = data[:, n]
    params = params[:, n]
    return data, params

def create_bootstrap_samples_and_predictions(
    dataset: Union[str, List[str]],
    region: str,
    model_type: Union[str, List[str]],
    metric: Union[str, List[str]],
    stat: Union[str, List[str]],
    n_samples: int = DEFAULT_N_SAMPLES) -> None:
    create_bootstrap_samples(dataset, region, model_type, metric, stat, n_samples=n_samples)
    create_bootstrap_predictions(dataset, region, model_type, metric, stat, n_samples=n_samples)

def __residuals(f):
    def inner(params, x, y, weights):
        y_pred = __f(x, params)
        rs = y - y_pred
        if weights is not None:
            rs *= weights
        return rs
    return inner

def __f(
    x: np.ndarray,
    params: Tuple[float]) -> Union[float, List[float]]:
    return -params[0] / (x - params[1]) + params[2]

def __get_p_init(metric: str) -> Tuple[float, float, float]:
    if higher_is_better(metric):
        return (1, -1, 1)
    else:
        return (-1, -1, 1)

def __fit_curve(
    p_init: Tuple[float],
    x: List[float],
    y: List[float], 
    max_nfev: int = DEFAULT_MAX_NFEV, 
    raise_error: bool = True, 
    weights: Optional[List[float]] = None):
    # Make fit.
    x_min = np.min(x)
    result = least_squares(__residuals(__f), p_init, args=(x, y, weights), bounds=((-np.inf, -np.inf, 0), (np.inf, x_min, np.inf)), max_nfev=max_nfev)

    # Check fit status.
    status = result.status
    if raise_error and status < 1:
        raise ValueError(f"Curve fit failed")

    # Check final parameter values.
    p_opt = result.x
    if raise_error: 
        if p_opt[1] >= x_min:
            raise ValueError(f"Vertical asymp. position 'x={p_opt[1]:.3f}' was right of minimum x value 'x={x_min}'. Bad 'p_init'?")
        elif p_opt[2] < 0:
            raise ValueError(f"Horizontal asymp. position 'y={p_opt[2]:.3f}' was below zero.")

    return p_opt, result.cost, result.jac

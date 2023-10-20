import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FixedLocator
from mymi.regions.tolerances import get_region_tolerance
import numpy as np
import os
import pandas as pd
from scipy.optimize import least_squares
import seaborn as sns
from tqdm import tqdm
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from mymi import config
from mymi.evaluation.dataset.nifti import load_segmenter_evaluation
from mymi.loaders import Loader, get_n_train_max
from mymi import logging
from mymi.metrics import higher_is_better
from mymi.regions import RegionNames
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

def __bootstrap_n_train_sample(x, n_samples, seed=42):
    np.random.seed(seed)
    return np.random.choice(x, size=(len(x), n_samples), replace=True)

def create_bootstrap_predictions(
    dataset: Union[str, List[str]],
    region: str,
    model_type: str,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    n_samples: int = DEFAULT_N_SAMPLES,
    raise_error: bool = True,
    weights: bool = False) -> None:
    datasets = arg_to_list(dataset, str)
    if '{tol}' in metric:
        metric = metric.replace('{tol}', str(get_region_tolerance(region)))
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))
    logging.arg_log('Creating bootstrap predictions', ('datasets', 'region', 'model_type', 'metric', 'stat'), (datasets, region, model_type, metric, stat))

    # Load samples.
    samples, n_trains = load_bootstrap_samples(datasets, region, model_type, metric, stat, include_n_trains=True, n_samples=n_samples)

    # Get placeholders.
    n_preds = np.max(n_trains) + 1
    n_samples = len(samples)
    n_params = 3
    params = np.zeros((n_samples, n_params))
    preds = np.zeros((n_samples, n_preds))

    # Get weights.
    weights = 1 / np.var(samples, axis=0).mean(axis=1) if weights else None

    for i in tqdm(range(n_samples)):
        sample = samples[i, :, :]
        
        # Flatten data.
        x = np.array([])
        y = np.array([])
        w = np.array([]) if weights is not None else None
        for j, (n_train, n_train_sample) in enumerate(zip(n_trains, sample)):
            x = np.concatenate((x, n_train * np.ones(len(n_train_sample))))
            y = np.concatenate((y, n_train_sample))
            if weights is not None:
                w = np.concatenate((w, weights[j] * np.ones(len(n_train_sample))))
            
        # Fit curve.
        try:
            p_init = __get_p_init(metric)
            p_opt, _, _ = __fit_curve(p_init, x, y, weights=w)
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
    filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'preds', encode(datasets), region, model_type, metric, stat, f'samples-{n_samples}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=preds, params=params)
    logging.info('Bootstrap predictions created.')

def create_bootstrap_samples(
    dataset: Union[str, List[str]],
    region: str, 
    model_type: Union[str, List[str]], 
    metric: Union[str, List[str]], 
    stat: Union[str, List[str]],
    n_samples: int = DEFAULT_N_SAMPLES) -> None:
    datasets = arg_to_list(dataset, str)
    model_types = arg_to_list(model_type, str)
    metrics = arg_to_list(metric, str)
    stats = arg_to_list(stat, str)
    logging.arg_log('Creating bootstrap samples', ('dataset', 'region', 'model_type', 'metric', 'stat'), (datasets, region, model_type, metric, stat))

    for model_type in model_types:
        for metric in metrics:
            for stat in stats:
                if '{tol}' in metric:
                    metric = metric.replace('{tol}', str(get_region_tolerance(region)))

                # Load all 'model_type' data.
                df = load_evaluation_data(datasets, region, model_type)
                if stat == 'mean':
                    data = get_mean_evaluation_data(df)
                elif stat == 'q1':
                    data = get_q1_evaluation_data(df)
                elif stat == 'q3':
                    data = get_q3_evaluation_data(df)

                # Bootstrap each 'n_train=...' sample to create a 3D array of 'n_samples' samples for each 'n_train'.
                boot_df = data[(data.metric == metric) & (data['model-type'] == model_type) & (data.region == region)]
                boot_df = boot_df.pivot(index=['region', 'model-type', 'n-train', 'metric'], columns='fold', values='value')
                boot_data = np.moveaxis(np.apply_along_axis(lambda x: __bootstrap_n_train_sample(x, n_samples), arr=boot_df.values, axis=1), 2, 0)
                n_trains = boot_df.reset_index()['n-train'].values
                
                # Save data.
                filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'samples', encode(datasets), region, model_type, metric, stat, f'samples-{n_samples}.npz')
                if not os.path.exists(os.path.dirname(filepath)):
                    os.makedirs(os.path.dirname(filepath))
                np.savez_compressed(filepath, data=boot_data, n_trains=n_trains)
                logging.info(f'Bootstrap samples created.')

def create_bootstrap_samples_and_predictions(
    dataset: Union[str, List[str]],
    region: str,
    model_type: Union[str, List[str]],
    metric: Union[str, List[str]],
    stat: Union[Literal['mean', 'q1', 'q3'], List[Literal['mean', 'q1', 'q3']]],
    n_samples: int = DEFAULT_N_SAMPLES) -> None:
    datasets = arg_to_list(dataset, str)
    model_types = arg_to_list(model_type, str)
    metrics = arg_to_list(metric, str)
    stats = arg_to_list(stat, str)
    arg_assert_literal_list(stats, str, ('mean', 'q1', 'q3'))
    logging.arg_log('Creating bootstrap samples and predictions', ('dataset', 'region', 'model_type', 'metric', 'stat'), (datasets, region, model_types, metrics, stats))

    # Create samples and prediction from curve fitting.
    for model_type in model_types:
        for metric in metrics:
            for stat in stats:
                create_bootstrap_samples(datasets, region, model_type, metric, stat, n_samples=n_samples)
                create_bootstrap_predictions(datasets, region, model_type, metric, stat, n_samples=n_samples)

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

def load_bootstrap_predictions(
    dataset: Union[str, List[str]],
    region: str,
    model_type: str,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    include_params: bool = False,
    n_samples: int = DEFAULT_N_SAMPLES) -> Tuple[np.ndarray]:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'preds', encode(datasets), region, model_type, metric, stat, f'samples-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    if include_params:
        params = f['params']
        return data, params
    else:
        return data

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
    data = load_bootstrap_predictions(dataset, region, model_type, metric, stat, include_params=include_params, n_samples=n_samples)
    if n == 'all':
        n = -1
    data = data[:, n]
    return data

def load_bootstrap_samples(
    dataset: Union[str, List[str]],
    region: str,
    model_type: str,
    metric: str,
    stat: Literal['mean', 'q1', 'q3'],
    include_n_trains: bool = False,
    n_samples: int = DEFAULT_N_SAMPLES):
    datasets = arg_to_list(dataset, str)

    filepath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'samples', encode(datasets), region, model_type, metric, stat, f'samples-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    if include_n_trains:
        n_trains = f['n_trains']
        return data, n_trains
    else:
        return data

def load_bootstrap_point_to_point_differences(
    dataset: Union[str, List[str]],
    region: str,
    model_type_a: str,
    model_type_b: str,
    n_train_a: int,
    n_train_b: int,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    n_samples: int = DEFAULT_N_SAMPLES) -> np.ndarray:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    # Calculate fitted value differences.
    preds_a = load_bootstrap_point_predictions(datasets, region, model_type_a, n_train_a, metric, stat, n_samples=n_samples)
    preds_b = load_bootstrap_point_predictions(datasets, region, model_type_b, n_train_b, metric, stat, n_samples=n_samples)
    diffs = preds_a - preds_b

    return diffs
    
def load_bootstrap_point_to_point_significant_differences(
    dataset: Union[str, List[str]],
    region: str,
    model_type_a: str,
    model_type_b: str,
    n_train_a: int,
    n_train_b: int,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    n_samples: int = DEFAULT_N_SAMPLES,
    p: float = 0.05) -> Tuple[int, float, Dict[int, str]]:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    # Load differences.
    diffs = load_bootstrap_point_to_point_differences(datasets, region, model_type_a, model_type_b, n_train_a, n_train_b, metric, stat, n_samples=n_samples)

    # Calculate stats.
    diff_mean = diffs.mean()
    diff_p5 = np.quantile(diffs, p)
    diff_p95 = np.quantile(diffs, 1 - p)

    # Create info dict.
    info = {
        0: model_type_a,
        1: model_type_b,
        2: 'No significant difference'
    }

    # Determine best model.
    if higher_is_better(metric): 
        if diff_p5 > 0:
            best_model = 0
        elif diff_p95 < 0:
            best_model = 1
        else:
            best_model = 2
    else:
        if diff_p5 > 0:
            best_model = 1
        elif diff_p95 < 0:
            best_model = 0
        else:
            best_model = 2

    # Get effect size.
    effect_size = diff_mean if best_model != 2 else np.nan

    return best_model, effect_size, info

def load_bootstrap_point_differences(
    dataset: Union[str, List[str]],
    region: str,
    model_type_a: str,
    model_type_b: str,
    n_train_a: int,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    n_samples: int = DEFAULT_N_SAMPLES) -> np.ndarray:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    # Calculate fitted value differences.
    preds_a = load_bootstrap_point_predictions(datasets, region, model_type_a, n_train_a, metric, stat, n_samples=n_samples)
    preds_b = load_bootstrap_predictions(datasets, region, model_type_b, metric, stat, n_samples=n_samples)
    preds_a = np.expand_dims(preds_a, 1).repeat(preds_b.shape[1], axis=1)
    diffs = preds_a - preds_b

    return diffs
    
def load_bootstrap_point_significant_differences(
    dataset: Union[str, List[str]],
    region: str,
    model_type_a: str,
    model_type_b: str,
    n_train_a: int,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    n_samples: int = DEFAULT_N_SAMPLES,
    p: float = 0.05) -> Tuple[int, float, Dict[int, str]]:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    # Load differences.
    diffs = load_bootstrap_point_differences(datasets, region, model_type_a, model_type_b, n_train_a, metric, stat, n_samples=n_samples)

    # Calculate stats.
    diff_means = diffs.mean(axis=0)
    diff_p5s = np.quantile(diffs, p, axis=0)
    diff_p95s = np.quantile(diffs, 1 - p, axis=0)

    # Create info dict.
    info = {
        0: model_type_a,
        1: model_type_b,
        2: 'No significant difference'
    }

    # Determine best models.
    best_models = np.full_like(diff_means, np.nan)
    if higher_is_better(metric): 
        best_models[diff_p5s > 0] = 0
        best_models[diff_p95s < 0] = 1
    else:
        best_models[diff_p5s > 0] = 1
        best_models[diff_p95s < 0] = 0
    best_models[(diff_p5s <= 0) & (diff_p95s >= 0)] = 2

    # Get effect sizes.
    effect_sizes = diff_means
    effect_sizes[best_models == 2] = np.nan

    return best_models, effect_sizes, info
    
def load_bootstrap_differences(
    dataset: Union[str, List[str]],
    region: str,
    model_type_a: str,
    model_type_b: str,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    n_samples: int = DEFAULT_N_SAMPLES) -> np.ndarray:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    # Calculate fitted value differences.
    preds_a = load_bootstrap_predictions(datasets, region, model_type_a, metric, stat, n_samples=n_samples)
    preds_b = load_bootstrap_predictions(datasets, region, model_type_b, metric, stat, n_samples=n_samples)
    diffs = preds_a - preds_b
    return diffs

def load_bootstrap_significant_differences(
    dataset: Union[str, List[str]],
    region: str,
    model_type_a: str,
    model_type_b: str,
    metric: str, 
    stat: Literal['mean', 'q1', 'q3'],
    n_samples: int = DEFAULT_N_SAMPLES,
    p: float = 0.05) -> Tuple[List[int], List[float], Dict[int, str]]:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    # Load differences.
    diffs = load_bootstrap_differences(datasets, region, model_type_a, model_type_b, metric, stat, n_samples=n_samples)

    # Calculate stats.
    diff_means = diffs.mean(axis=0)
    diff_p5s = np.quantile(diffs, p, axis=0)
    diff_p95s = np.quantile(diffs, 1 - p, axis=0)

    # Create info dict.
    info = {
        0: model_type_a,
        1: model_type_b,
        2: 'No significant difference'
    }

    # Determine best models.
    best_models = np.full_like(diff_means, np.nan)
    if higher_is_better(metric): 
        best_models[diff_p5s > 0] = 0
        best_models[diff_p95s < 0] = 1
    else:
        best_models[diff_p5s > 0] = 1
        best_models[diff_p95s < 0] = 0
    best_models[(diff_p5s <= 0) & (diff_p95s >= 0)] = 2

    # Get effect sizes.
    effect_sizes = diff_means
    effect_sizes[best_models == 2] = np.nan

    return best_models, effect_sizes, info

def load_evaluation_data(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model_type: Union[Literal['clinical', 'transfer'], List[Literal['clinical', 'transfer']]],
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

def get_mean_evaluation_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['fold', 'region', 'model-type', 'n-train', 'metric'])['value'].mean().reset_index()

def get_q1_evaluation_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['fold', 'region', 'model-type', 'n-train', 'metric'])['value'].quantile(0.25).reset_index()

def get_q3_evaluation_data(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['fold', 'region', 'model-type', 'n-train', 'metric'])['value'].quantile(0.75).reset_index()

def megaplot(
    dataset: Union[str, List[str]],
    region: Union[str, List[str]],
    model: Union[str, List[str]],
    metric: Union[str, List[str], np.ndarray],
    stat: Union[str, List[str]],
    figsize: Optional[Tuple[float, float]] = None,
    fontsize: float = DEFAULT_FONT_SIZE,
    fontsize_tick_label: Optional[float] = None,
    height_ratios: Optional[List[float]] = None,
    hspace_grid: float = 0.25,
    hspace_plot: float = 0.25,
    hspace_plot_xlabel: float = 0.5,
    legend_loc: Optional[Union[str, List[str]]] = None,
    model_label: Optional[Union[str, List[str]]] = None,
    savepath: Optional[str] = None,
    secondary_stat: Optional[Union[str, List[str], np.ndarray]] = None,
    wspace: float = 0.25,
    y_lim: bool = True,
    **kwargs: Dict[str, Any]) -> None:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    n_regions = len(regions)
    if height_ratios is not None:
        assert len(height_ratios) == n_regions
    if type(metric) is str:
        metrics = np.repeat([[metric]], n_regions, axis=0)
    elif type(metric) is list:
        metrics = np.repeat([metric], n_regions, axis=0)
    else:
        metrics = metric
    if type(secondary_stat) is str:
        secondary_stats = np.repeat([[secondary_stat]], n_regions, axis=0)
    elif type(secondary_stat) is list:
        secondary_stats = np.repeat([secondary_stat], n_regions, axis=0)
    else:
        secondary_stats = secondary_stat

    n_metrics = metrics.shape[1]
    legend_locs = arg_to_list(legend_loc, str)
    if legend_locs is not None:
        assert len(legend_locs) == n_metrics
    models = arg_to_list(model, str)
    n_models = len(models)
    model_labels = arg_to_list(model_label, str)
    if model_labels is not None:
        assert len(model_labels) == n_models
    stats = arg_broadcast(stat, n_metrics, str)

    # Lookup tables.
    # matplotlib.rc('text', usetex=True)
    # matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # Create main gridspec (size: num OARs x num metrics).
    if figsize is None:
        figsize=(6 * metrics.shape[1], 6 * n_regions)
    fig = plt.figure(constrained_layout=False, figsize=figsize)
    gs = GridSpec(n_regions, metrics.shape[1], figure=fig, height_ratios=height_ratios, hspace=hspace_grid, wspace=wspace)
    for i, region in enumerate(regions):
        for j in range(n_metrics):
            metric = metrics[i, j]
            stat = stats[j]
            sec_stat = secondary_stats[i, j] if secondary_stats is not None else None
            x_label = 'num. institutional samples (n)' if i == len(regions) - 1 else None
            y_lim = DEFAULT_METRIC_Y_LIMS[metric] if y_lim else (None, None)
            legend_loc = legend_locs[j] if legend_locs is not None else DEFAULT_METRIC_LEGEND_LOCS[metric]
            plot_bootstrap_fit(datasets, region, models, metric, stat, fontsize=fontsize, fontsize_tick_label=fontsize_tick_label, hspace=hspace_plot, hspace_xlabel=hspace_plot_xlabel, legend_loc=legend_loc, model_labels=model_labels, outer_gs=gs[i, j], secondary_stat=sec_stat, split=True, x_label=x_label, x_scale='log', y_label=DEFAULT_METRIC_LABELS[metric], y_lim=y_lim, **kwargs)

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

    size = fig.get_size_inches() * fig.dpi
    plt.show()

def plot_bootstrap_fit(
    dataset: Union[str, List[str]],
    region: str, 
    model_type: Union[str, List[str]],
    metric: str,
    stat: Literal['mean', 'q1', 'q3'],
    alpha_ci: float = 0.2,
    alpha_points: float = 1.0,
    alpha_secondary: float = 0.5,
    ax: Optional[Union[mpl.axes.Axes, List[mpl.axes.Axes]]] = None,
    diff_hspace: Optional[float] = 0.05,
    diff_marker: str = 's',
    figsize: Tuple[float, float] = (8, 6),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontsize_label: Optional[float] = None,
    fontsize_tick_label: Optional[float] = None,
    fontweight: Literal['normal', 'bold'] = 'normal',
    hspace: float = 0,
    hspace_xlabel: float = 0.2,
    labelpad: float = 0,
    legend_loc: Optional[str] = None,
    linewidth: float = 1,
    model_labels: Optional[Union[str, List[str]]] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    point_size: float = 1,
    secondary_stat: Optional[str] = None,
    show_ci: bool = True,
    show_diff: bool = True,
    show_legend: bool = True,
    show_limits: bool = False,
    show_points: bool = True,
    show_public_colour: bool = True,
    show_secondary_stat_ci: bool = False, 
    show_secondary_stat_diff: bool = True,
    split: bool = True,
    split_wspace: Optional[float] = 0.05,
    outer_gs: Optional[mpl.gridspec.SubplotSpec] = None,
    ticklength: float = 1,
    tickpad: float = 2,
    title: str = '',
    titlepad: float = 2,
    x_label: Optional[str] = None,
    x_scale: str = 'log',
    y_label: Optional[str] = None,
    y_lim: Optional[Tuple[float, float]] = None):
    datasets = arg_to_list(dataset, str)
    model_types = arg_to_list(model_type, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))
    axs = arg_to_list(ax, mpl.axes.Axes)
    n_colours = len(model_types) + 1 if show_public_colour else len(model_types)
    model_colours = sns.color_palette('colorblind')[:n_colours]
    fontsize_label = fontsize if fontsize_label is None else fontsize_label
    fontsize_tick_label = fontsize if fontsize_tick_label is None else fontsize_tick_label
    legend_loc = DEFAULT_METRIC_LEGEND_LOCS[metric] if legend_loc is None else legend_loc
    if model_labels is not None:
        model_labels = [model_labels] if type(model_labels) == str else model_labels
        assert len(model_labels) == len(model_types)
    if secondary_stat is None:
        show_secondary_stat_diff = False
        
    # Create gridspec. If using 'megaplot' this sits inside the large megaplot gridspec.
    # Size = (2 x 1) if showing significant differences, otherwise (1 x 1). Both primary and secondary
    # stat significant differences share the bottom row.
    gs_size = (2, 1) if show_diff else (1, 1)
    gs_height_ratios = (48 if show_secondary_stat_diff else 49, 2 if show_secondary_stat_diff else 1) if show_diff else None
    hspace = hspace_xlabel if x_label is not None else hspace
    if outer_gs is None:
        # Create standalone gridspec for 'bootstrap fit'.
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(*gs_size, figure=fig, height_ratios=gs_height_ratios, hspace=hspace)
    else:
        # Nest 'bootstrap fit' gridspec in 'megaplot' gridspec.
        fig = outer_gs.get_gridspec().figure
        gs = GridSpecFromSubplotSpec(*gs_size, subplot_spec=outer_gs, height_ratios=gs_height_ratios, hspace=hspace)

    # Create data gridspec.
    # Size = (1 x 2) if splitting, otherwise (1, 1).
    data_gs_size = (1, 2) if split else (1, 1)
    data_gs_width_ratios = (1, 19) if split else None
    data_gs = GridSpecFromSubplotSpec(*data_gs_size, subplot_spec=gs[0, 0], width_ratios=data_gs_width_ratios, wspace=split_wspace)

    # Create difference gridspec.
    if show_diff:
        if split:
            diff_gs_size = (2, 2) if show_secondary_stat_diff else (1, 2)
        else:
            diff_gs_size = (2, 1) if show_secondary_stat_diff else (1, 1)
        diff_gs_hspace = diff_hspace if show_secondary_stat_diff else None
        diff_gs_width_ratios = (1, 19) if split else None
        diff_gs = GridSpecFromSubplotSpec(*diff_gs_size, hspace=diff_gs_hspace, subplot_spec=gs[1, 0], width_ratios=diff_gs_width_ratios, wspace=split_wspace)

    # Create subplot axes.
    # The dimensions of these lists will change depending upon the options:
    #   'split', 'show_diff' and 'show_secondary_stat_diff'. 
    y_axis = fig.add_subplot(data_gs[0, 0])
    axs = [
        [
            y_axis,
            # Add this axis if we're splitting horizontally. Public results (n=0) will be shown on left split,
            # all other results (n=5,10,...) will be shown on right split.
            *([fig.add_subplot(data_gs[0, 1], sharey=y_axis)] if split else []),
        ],
        # Add this/these axes if we're showing significant differences below the plot. This is another plot
        # that's very thin vertically.
        *([
            [
                # If we're split, don't show significant differences on the left split as it's just the public
                # model results here.
                *([None] if split else []),
                fig.add_subplot(diff_gs[0, 1] if split else diff_gs[0, 0])
            ]
        ] if show_diff else []),
        # Add these axes if we're showing secondary statistic (e.g. Q1/Q3) significant differences.
        *([
            [
                *([None] if split else []),
                fig.add_subplot(diff_gs[1, 1] if split else diff_gs[1, 0])
            ]
        ] if show_diff and show_secondary_stat_diff else [])
    ]
    
    # Plot main data.
    for ax in axs[0]:
        if ax is None:
            continue

        for i, model_type in enumerate(model_types):
            model_colour = model_colours[i]
            model_label = model_labels[i] if model_labels is not None else model_type

            # Load bootstrapped predictions.
            preds = load_bootstrap_predictions(datasets, region, model_type, metric, stat, n_samples=n_samples)

            # Load data for secondary statistic.
            if secondary_stat:
                sec_preds = load_bootstrap_predictions(datasets, region, model_type, metric, secondary_stat, n_samples=n_samples)

            # Plot original data (before bootstrapping was applied).
            if show_points:
                x_raw, y_raw = raw_data(datasets, region, model_type, metric, stat)
                if model_type == 'transfer' and show_public_colour:
                    ind_0 = list(np.argwhere(np.array(x_raw) == 0).flatten())
                    x_raw_0 = [x for i, x in enumerate(x_raw) if i in ind_0]
                    y_raw_0 = [y for i, y in enumerate(y_raw) if i in ind_0]
                    x_raw_other = [x for i, x in enumerate(x_raw) if i not in ind_0]
                    y_raw_other = [y for i, y in enumerate(y_raw) if i not in ind_0]
                    ax.scatter(x_raw_0, y_raw_0, alpha=alpha_points, color=model_colours[-1], edgecolors='none', marker='^', s=point_size)
                    ax.scatter(x_raw_other, y_raw_other, alpha=alpha_points, color=model_colour, marker='x', s=point_size, linewidth=linewidth)
                else:
                    ax.scatter(x_raw, y_raw, alpha=alpha_points, color=model_colour, edgecolors='none', marker='s', s=point_size)

            # Plot mean value of 'stat' over all bootstrapped samples (convergent value).
            means = preds.mean(axis=0)
            x = np.linspace(0, len(means) - 1, num=len(means))
            if model_type == 'transfer' and show_public_colour:
                ax.plot(x[:2], means[:2], color=model_colours[-1], label=model_label, linewidth=linewidth, linestyle='solid')
                ax.plot(x[1:], means[1:], color=model_colour, label=model_label, linewidth=linewidth, linestyle='solid')
            else:
                ax.plot(x, means, color=model_colour, label=model_label, linewidth=linewidth, linestyle='solid')

            # Plot secondary statistic mean values.
            if secondary_stat:
                sec_means = sec_preds.mean(axis=0)
                if model_type == 'transfer' and show_public_colour:
                    ax.plot(x[:2], sec_means[:2], color=model_colours[-1], alpha=alpha_secondary, linestyle='dashed', linewidth=linewidth)
                    ax.plot(x[1:], sec_means[1:], color=model_colour, alpha=alpha_secondary, linestyle='dashed', linewidth=linewidth)
                else:
                    ax.plot(x, sec_means, color=model_colour, alpha=alpha_secondary, linestyle='dashed', linewidth=linewidth)

            # Plot 95% CIs for statistic.
            if show_ci:
                low_ci = np.quantile(preds, 0.025, axis=0)
                high_ci = np.quantile(preds, 0.975, axis=0)
                if model_type == 'transfer' and show_public_colour:
                    ax.fill_between(x[:2], low_ci[:2], high_ci[:2], alpha=alpha_ci, color=model_colours[-1], linewidth=linewidth)
                    ax.fill_between(x[1:], low_ci[1:], high_ci[1:], alpha=alpha_ci, color=model_colour, linewidth=linewidth)
                else:
                    ax.fill_between(x, low_ci, high_ci, alpha=alpha_ci, color=model_colour, linewidth=linewidth)

            # Plot secondary statistic 95% CIs.
            if secondary_stat and show_secondary_stat_ci:
                low_ci = np.quantile(sec_preds, 0.025, axis=0)
                high_ci = np.quantile(sec_preds, 0.975, axis=0)
                if model_type == 'transfer' and show_public_colour:
                    ax.fill_between(x[:2], low_ci[:2], high_ci[:2], alpha=alpha_secondary * alpha_ci, color=model_colours[-1], linewidth=linewidth)
                    ax.fill_between(x[1:], low_ci[1:], high_ci[1:], alpha=alpha_secondary * alpha_ci, color=model_colour, linewidth=linewidth)
                else:
                    ax.fill_between(x, low_ci, high_ci, alpha=alpha_secondary * alpha_ci, color=model_colour, linewidth=linewidth)

            # Plot upper/lower limits for statistic.
            if show_limits:
                min = preds.min(axis=0)
                max = preds.max(axis=0)
                ax.plot(x, min, c='black', linestyle='--', alpha=0.5, linewidth=linewidth)
                ax.plot(x, max, c='black', linestyle='--', alpha=0.5, linewidth=linewidth)

    # Plot difference.
    if show_diff:
        assert len(model_types) == 2

        # Load significant difference.
        best_models, _, _ = load_bootstrap_significant_differences(datasets, region, *model_types, metric, stat, n_samples=n_samples)
        
        # Plot model A points.
        diff_ax = axs[1][1] if split else axs[1][0]
        x = np.argwhere(best_models == 0).flatten()
        diff_ax.scatter(x, [0] * len(x), color=model_colours[0], marker=diff_marker, s=point_size)

        # Plot model B points.
        x = np.argwhere(best_models == 1).flatten()
        diff_ax.scatter(x, [0] * len(x), color=model_colours[1], marker=diff_marker, s=point_size)

        if show_secondary_stat_diff:
            # Load significant difference.
            best_models, _, _ = load_bootstrap_significant_differences(datasets, region, *model_types, metric, secondary_stat, n_samples=n_samples)
            
            # Plot model A points.
            diff_ax = axs[2][1] if split else axs[2][0]
            x = np.argwhere(best_models == 0).flatten()
            diff_ax.scatter(x, [0] * len(x), color=model_colours[0], marker=diff_marker, s=point_size)

            # Plot model B points.
            x = np.argwhere(best_models == 1).flatten()
            diff_ax.scatter(x, [0] * len(x), color=model_colours[1], marker=diff_marker, s=point_size)

    # Set axis scale.
    if split:
        axs[0][1].set_xscale(x_scale)

        if show_diff:
            axs[1][1].set_xscale(x_scale)

            if show_secondary_stat_diff:
                axs[2][1].set_xscale(x_scale)
    else:
        axs[0][0].set_xscale(x_scale)

        if show_diff:
            axs[1][0].set_xscale(x_scale)

            if show_secondary_stat_diff:
                axs[2][0].set_xscale(x_scale)

    # Set x/y label.
    if x_label is not None:
        x_axis = axs[0][1] if split else axs[0][0]
        x_axis.set_xlabel(x_label, fontsize=fontsize_label, labelpad=labelpad, weight=fontweight)
    if y_label is not None:
        axs[0][0].set_ylabel(y_label, fontsize=fontsize_label, labelpad=labelpad, weight=fontweight)

    # Set y limits.
    axs[0][0].set_ylim(y_lim)
    if split:
        axs[0][1].set_ylim(y_lim)

    # Set x tick labels.
    x_upper_lim = None
    if split:
        axs[0][0].set_xticks([0])
        if show_diff:
            axs[1][1].get_xaxis().set_visible(False)
            axs[1][1].get_yaxis().set_visible(False)

            if show_secondary_stat_diff:
                axs[2][1].get_xaxis().set_visible(False)
                axs[2][1].get_yaxis().set_visible(False)

        if x_scale == 'log':
            x_upper_lim = axs[0][1].get_xlim()[1]      # Record x upper lim as setting ticks overrides this.
            axs[0][1].set_xticks(LOG_SCALE_X_TICK_LABELS)
            axs[0][1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    else:
        if show_diff:
            axs[1][0].get_xaxis().set_visible(False)
            axs[1][0].get_yaxis().set_visible(False)

            if show_secondary_stat_diff:
                axs[2][0].get_xaxis().set_visible(False)
                axs[2][0].get_yaxis().set_visible(False)

    # Set axis limits.
    if split:
        axs[0][0].set_xlim(-0.5, 0.5)
        axs[0][1].set_xlim(4.5, x_upper_lim)

        if show_diff:
            axs[1][1].set_xlim(4.5, x_upper_lim)

            if show_secondary_stat_diff:
                axs[2][1].set_xlim(4.5, x_upper_lim)

    # Set x/y tick label style.
    axs[0][0].tick_params(axis='x', which='major', labelsize=fontsize_tick_label, length=ticklength, pad=tickpad, width=linewidth)
    axs[0][0].tick_params(axis='x', which='minor', direction='in', length=ticklength, width=linewidth)
    axs[0][0].tick_params(axis='y', which='major', labelsize=fontsize_tick_label, length=ticklength, pad=tickpad, width=linewidth)
    if split:
        axs[0][1].tick_params(axis='x', which='major', labelsize=fontsize_tick_label, length=ticklength, pad=tickpad, width=linewidth)
        axs[0][1].tick_params(axis='x', which='minor', direction='in', length=ticklength, width=linewidth)
        axs[0][1].tick_params(axis='y', color='white', labelleft=False)      # Hide split's y axis labels and ticks. Can't remove as grid lines disappear.
        # axs[0][1].tick_params(color='white', labelleft=False)   # Can't remove y tick params, as gridlines will disappear too.

    # Set spine linewidths.
    spines = ['top', 'bottom','left','right']
    for spine in spines:
        axs[0][0].spines[spine].set_linewidth(linewidth)
    if split:
        for spine in spines:
            axs[0][1].spines[spine].set_linewidth(linewidth)

        if show_diff:
            for spine in spines:
                axs[1][1].spines[spine].set_linewidth(linewidth)
            if show_secondary_stat_diff:
                for spine in spines:
                    axs[2][1].spines[spine].set_linewidth(linewidth)
    else:
        if show_diff:
            for spine in spines:
                axs[1][0].spines[spine].set_linewidth(linewidth)
            if show_secondary_stat_diff:
                for spine in spines:
                    axs[2][0].spines[spine].set_linewidth(linewidth)

    # Set title.
    title = title if title else region
    title_axis = axs[0][1] if split else axs[0][0]
    title_axis.set_title(title, fontsize=fontsize, pad=titlepad, weight=fontweight)

    # Add legend.
    if show_legend:
        if split:
            axs[0][1].legend(fontsize=fontsize, loc=legend_loc)
        else:
            axs[0][0].legend(fontsize=fontsize, loc=legend_loc)

    # Hide axis spines.
    if split:
        axs[0][0].spines['right'].set_visible(False)
        axs[0][1].spines['left'].set_visible(False)

    # Add split between axes.
    if split:
        d_x_0 = .1
        d_x_1 = .006
        d_y = .03
        kwargs = dict(transform=axs[0][0].transAxes, color='k', clip_on=False)
        axs[0][0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (-d_y / 2, d_y / 2), linewidth=linewidth, **kwargs)  # bottom-left diagonal
        axs[0][0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (1 - (d_y / 2), 1 + (d_y / 2)), linewidth=linewidth, **kwargs)  # top-left diagonal
        kwargs = dict(transform=axs[0][1].transAxes, color='k', clip_on=False)
        axs[0][1].plot((-d_x_1 / 2, d_x_1 / 2), (-d_y / 2, d_y / 2), linewidth=linewidth, **kwargs)  # bottom-left diagonal
        axs[0][1].plot((-d_x_1 / 2, d_x_1 / 2), (1 - (d_y / 2), 1 + (d_y / 2)), linewidth=linewidth, **kwargs)  # top-left diagonal

    # Apply formatting (e.g 1k, 2k, etc.) if all values are over 1000.
    y_tick_locs = axs[0][0].get_yticks()
    apply_formatting = True
    for y_tick_loc in y_tick_locs:
        if y_tick_loc != 0 and y_tick_loc < 1000:
            apply_formatting = False
    if apply_formatting:
        def format_y_tick(y_tick_loc: float):
            y_tick = int(y_tick_loc / 1000)
            return f'{y_tick}k'
        axs[0][0].yaxis.set_major_locator(FixedLocator(y_tick_locs))
        axs[0][0].set_yticklabels([format_y_tick(y) for y in y_tick_locs])

    # Add gridlines.
    axs[0][0].grid(which='major', axis='x', color='grey', alpha=0.1, linewidth=0.5)
    axs[0][0].grid(which='major', axis='y', color='grey', alpha=0.1, linewidth=0.5)
    if split:
        axs[0][1].grid(which='major', axis='x', color='grey', alpha=0.1, linewidth=0.5)
        axs[0][1].grid(which='major', axis='y', color='grey', alpha=0.1, linewidth=0.5)

def raw_data(
    dataset: Union[str, List[str]],
    region: str,
    model_type: str,
    metric: str,
    stat: Literal['mean', 'q1', 'q3']) -> Tuple[List[int], List[float]]:
    datasets = arg_to_list(dataset, str)
    arg_assert_literal(stat, ('mean', 'q1', 'q3'))

    # Load evaluation data.
    df = load_evaluation_data(datasets, region, model_type)
    if stat == 'mean':
        stat_df = get_mean_evaluation_data(df)
    elif stat == 'q1':
        stat_df = get_q1_evaluation_data(df)
    elif stat == 'q3':
        stat_df = get_q3_evaluation_data(df)

    # Get data points.
    data_df = stat_df[(stat_df.metric == metric) & (stat_df['model-type'] == model_type) & (stat_df.region == region)]
    x = data_df['n-train'].tolist()
    y = data_df['value'].tolist()
    return x, y

def __residuals(f):
    def inner(params, x, y, weights):
        y_pred = __f(x, params)
        rs = y - y_pred
        if weights is not None:
            rs *= weights
        return rs
    return inner

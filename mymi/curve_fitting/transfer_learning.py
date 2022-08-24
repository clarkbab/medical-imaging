import numpy as np
import os
import pandas as pd
from scipy.optimize import least_squares
from typing import Callable, List, Optional, Tuple

from mymi import config
from mymi.evaluation.dataset.nifti import load_segmenter_evaluation_from_loader
from mymi.loaders import Loader

def __bootstrap_n_train_sample(x, n_samples, seed=42):
    np.random.seed(seed)
    return np.random.choice(x, size=(len(x), n_samples), replace=True)

def create_bootstrap_predictions(
    data: pd.DataFrame, 
    metric: str, 
    model: str,
    n_trains: Tuple[int],
    region: str,
    raise_error=True,
    weights=True):
    # Determine 'p_init'.
    f = f1
    f_name = 'f1'
    if metric == 'dice' or 'surface-dice' in metric:
        p_init = p_init_inc1
    elif metric in []


    # Get placeholders.
    n_samples = len(data)
    n_preds = n_trains.max() - 4 if model == 'clinical' else n_trains.max() + 1
    params = np.zeros((n_samples, len(p_init)))
    preds = np.zeros((n_samples, n_preds))

    # Get weights.
    weights = 1 / np.var(data, axis=0).mean(axis=1) if weights else None

    for i in range(n_samples):
        sample = data[i, :, :]
        
        # Flatten data.
        x = np.array([])
        y = np.array([])
        w = np.array([]) if weights is not None else None
        for i, (n_train, n_train_sample) in enumerate(zip(n_trains, sample)):
            x = np.concatenate((x, n_train * np.ones(len(n_train_sample))))
            y = np.concatenate((y, n_train_sample))
            if weights is not None:
                w = np.concatenate((w, weights[i] * np.ones(len(n_train_sample))))
            
        # Fit curve.
        try:
            p_opt, _, _ = fit_curve(f, p_init, x, y, weights=w)
        except ValueError as e:
            if raise_error:
                raise e
            else:
                return x, y, w
        params[i] = p_opt
        
        # Create prediction points.
        n_train_min = 5 if model == 'clinical' else 0
        x = np.linspace(n_train_min, n_train_min + n_preds, num=n_preds)
        y = f(x, p_opt)
        preds[i] = y
        
    # Save data.
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'preds', f_name, model, metric)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=preds, params=params)

def create_bootstrap_samples(
    data: pd.DataFrame, 
    metric: str, 
    model: str, 
    region: str, 
    n_samples: int) -> None:
    # Bootstrap each 'n_train=...' sample to create a 3D array of 'n_samples' samples for each 'n_train'.
    boot_df = data[(data.metric == metric) & (data['model-type'] == model) & (data.region == region)]
    boot_df = boot_df.pivot(index=['region', 'model-type', 'n-train', 'metric'], columns='fold', values='mean-value')
    boot_data = np.moveaxis(np.apply_along_axis(lambda x: __bootstrap_n_train_sample(x, n_samples), arr=boot_df.values, axis=1), 2, 0)
    n_trains = boot_df.reset_index()['n-train'].values
    
    # Save data.
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'samples', model, metric)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=boot_data, n_trains=n_trains)

def create_bootstrap_samples_and_predictions(
    data: pd.DataFrame, 
    metric: str, 
    model: str, 
    region: str, 
    n_samples: int) -> None:

    # Create samples.
    create_bootstrap_samples(data, metric, model, region, n_samples)

    # Create predictions for fitted curves.
    create_bootstrap_predictions(data, f1, 'f1', f1_init)


def f1(
    x: np.ndarray,
    params: Tuple[float]):
    return 1 / (params[0] * x + params[1]) + params[2]
p_init_inc1 = (-0.5, -0.5, 0.5)
p_init_dec1 = (0.5, 0.5, 0.5)

def f2(
    x: np.ndarray,
    params: Tuple[float]):
    return np.exp(params[0] * x + params[1]) + params[2]
p_init_inc2 = (-0.5, -0.5, 0.5)
p_init_dec2 = (0.5, 0.5, 0.5)

def fit_curve(
    f: Callable, 
    p_init: Tuple[float], 
    x: List[float],
    y: List[float], 
    max_nfev: int = int(1e6), 
    raise_error: bool = True, 
    weights: Optional[List[float]] = None):
    result = least_squares(__residuals(f), p_init, args=(x, y, weights), max_nfev=max_nfev, method='lm')
    p_opt = result.x
    loss = result.cost
    jac = result.jac
    status = result.status
    if raise_error and status < 1:
        raise ValueError(f"Curve fit failed")
    return p_opt, loss, jac

def load_bootstrap_predictions(
    f_name: str, 
    metric: str, 
    model: str,
    region: str,
    n_samples: int):
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'preds', f_name, model, metric)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    params = f['params']
    return data, params

def load_bootstrap_samples(
    metric: str,
    model: str,
    region: str,
    n_samples: int):
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'curve-fitting', 'bootstrap', 'samples', model, metric)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    n_trains = f['n_trains']
    return data, n_trains

def load_mean_evaluation(
    datasets: Tuple[str],
    regions: Tuple[str],
    models: Tuple[str],
    num_trains: Tuple[Optional[int]],
    num_folds: int,
    test_folds: Tuple[int]) -> pd.DataFrame:
    dfs = []
    for region in regions:
        for test_fold in test_folds:
            # Add public evaluation.
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')
            seg_run = 'public-1gpu-150epochs'
            segmenter = (f'segmenter-{region}', seg_run, 'BEST')
            df = load_segmenter_evaluation_from_loader(datasets, localiser, segmenter, num_folds, test_fold)
            df['model-type'] = f'transfer-fold-{test_fold}-samples-0'
            dfs.append(df)

            # Get number of training cases.
            tl, vl, _ = Loader.build_loaders(datasets, region, num_folds=num_folds, test_fold=test_fold)
            n_train_max = len(tl) + len(vl)

            # Add clinical/transfer evaluations.
            for model in (model for model in models if model != 'public'):
                for num_train in num_trains:
                    # Check that number of training cases are available.
                    if num_train is not None and num_train > n_train_max:
                        continue

                    seg_run = f'{model}-fold-{test_fold}-samples-{num_train}'
                    segmenter = (f'segmenter-{region}', seg_run, 'BEST')
                    df = load_segmenter_evaluation_from_loader(datasets, localiser, segmenter, num_folds, test_fold)
                    df['model-type'] = seg_run
                    dfs.append(df)
                   
    # Save dataframe.
    df = pd.concat(dfs, axis=0)

    # Add num-train (same for each fold).
    df = df.assign(**{ 'n-train': df['model-type'].str.split('-').apply(lambda x: x[-1]) })
    none_models = df[df['model-type'].str.contains('-None')]['model-type'].unique()
    none_nums = {}
    for region in regions:
        tl, vl, _ = Loader.build_loaders(datasets, region, num_folds=num_folds, test_fold=0)
        n_train = len(tl) + len(vl)
        none_nums[region] = n_train
    df.loc[df['n-train'] == 'None', 'n-train'] = df[df['n-train'] == 'None']['region'].apply(lambda r: none_nums[r])
    df['n-train'] = df['n-train'].astype(int)

    # Replace model names.
    df['model-type'] = df['model-type'].str.split('-').apply(lambda l: l[0])
    df['model'] = df['model-type'] + '-' + df['n-train'].astype(str)

    # Sort values.
    df = df.sort_values(['fold', 'region', 'model-type', 'n-train'])

    # Add region counts.
    # count_df = df.groupby(['fold', 'region', 'model-type', 'metric'])['patient-id'].count().reset_index()

    # Get mean values.
    mean_df = df.groupby(['fold', 'region', 'model-type', 'n-train', 'metric'])['value'].mean().rename('mean-value').reset_index()

    return mean_df

def __residuals(f):
    def inner(params, x, y, weights):
        y_pred = f(x, params)
        residuals = y - y_pred
        if weights is not None:
            residuals *= weights
        return residuals
    return inner

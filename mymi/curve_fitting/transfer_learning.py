import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
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
from mymi.regions import RegionNames
from mymi.utils import arg_assert_present, arg_broadcast, arg_to_list

DEFAULT_FONT_SIZE = 15
DEFAULT_MAX_NFEV = int(1e6)
DEFAULT_METRIC_LEGEND_LOCS = {
    'dice': 'lower right',
    'hd': 'upper right',
    'hd-95': 'upper right',
    'msd': 'upper right'
}
DEFAULT_METRIC_LABELS = {
    'dice': 'DSC',
    'hd': 'HD [mm]',
    'hd-95': '95HD [mm]',
    'msd': 'MSD [mm]',
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
    DEFAULT_METRIC_LABELS[f'apl-mm-tol-{tol}'] = fr'APL, $\tau$={tol}mm'
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
    region: str,
    model: str,
    metric: str, 
    stat: str,
    samples: np.ndarray,
    n_trains: Union[int, List[int]],
    raise_error: bool = True,
    weights: bool = False) -> None:
    logging.info(f"Creating bootstrap predictions for metric '{metric}', model '{model}', stat '{stat}', n_trains '{n_trains}', region '{region}'...")

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
            p_init = get_p_init(metric)
            p_opt, _, _ = fit_curve(p_init, x, y, weights=w)
        except ValueError as e:
            if raise_error:
                logging.error(f"Error when fitting sample '{i}':")
                raise e
            else:
                return x, y, w
        params[i] = p_opt
        
        # Create prediction points.
        x = np.linspace(0, n_preds - 1, num=n_preds)
        y = f(x, p_opt)
        preds[i] = y

    # Set 'clinical' model predictions to 'NaN' as 'clinical' model must be trained on
    # 5 or more training cases.
    if model == 'clinical':
        preds[:, :5] = np.nan
        
    # Save data.
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'preds', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=preds, params=params)
    logging.info('Bootstrap predictions created.')

def create_bootstrap_samples(
    region: str, 
    model: str, 
    metric: str, 
    stat: str,
    data: pd.DataFrame, 
    n_samples: int = DEFAULT_N_SAMPLES) -> Tuple[np.ndarray, List[int]]:
    logging.info(f"Creating bootstrap samples for region '{region}', model '{model}', metric '{metric}', stat '{stat}', n_samples '{n_samples}'...")

    # Bootstrap each 'n_train=...' sample to create a 3D array of 'n_samples' samples for each 'n_train'.
    boot_df = data[(data.metric == metric) & (data['model-type'] == model) & (data.region == region)]
    boot_df = boot_df.pivot(index=['region', 'model-type', 'n-train', 'metric'], columns='fold', values='value')
    boot_data = np.moveaxis(np.apply_along_axis(lambda x: __bootstrap_n_train_sample(x, n_samples), arr=boot_df.values, axis=1), 2, 0)
    n_trains = boot_df.reset_index()['n-train'].values
    
    # Save data.
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'samples', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    np.savez_compressed(filepath, data=boot_data, n_trains=n_trains)
    logging.info(f'Bootstrap samples created.')

    return boot_data, list(n_trains)

def create_bootstrap_samples_and_predictions(
    datasets: List[str],
    region: str,
    model_types: Union[str, List[str]],
    metrics: Union[str, List[str]],
    stats: Union[str, List[str]],
    n_trains: Union[Optional[int], List[Optional[int]]] = DEFAULT_N_TRAINS,
    n_folds: int = 5,
    n_samples: int = DEFAULT_N_SAMPLES,
    test_folds: Union[int, List[int]] = list(range(5))) -> None:
    metrics = [metrics] if type(metrics) == str else metrics
    model_types = [model_types] if type(model_types) == str else model_types
    stats = [stats] if type(stats) == str else stats
    test_folds = [test_folds] if type(test_folds) == int else test_folds

    # Load data.
    df = load_evaluation_data(datasets, region, model_types, n_trains=n_trains, n_folds=n_folds, test_folds=test_folds)
    if 'mean' in stats:
        mean_df = get_mean_evaluation_data(df)
    if 'q1' in stats:
        q1_df = get_q1_evaluation_data(df)
    if 'q3' in stats:
        q3_df = get_q3_evaluation_data(df)

    # Create samples and prediction from curve fitting.
    for metric in metrics:
        for model_type in model_types:
            if 'mean' in stats:
                samples, n_trains = create_bootstrap_samples(region, model_type, metric, 'mean', mean_df, n_samples=n_samples)
                create_bootstrap_predictions(region, model_type, metric, 'mean', samples, n_trains)
            if 'q1' in stats:
                samples, n_trains = create_bootstrap_samples(region, model_type, metric, 'q1', q1_df, n_samples=n_samples)
                create_bootstrap_predictions(region, model_type, metric, 'q1', samples, n_trains)
            if 'q3' in stats:
                samples, n_trains = create_bootstrap_samples(region, model_type, metric, 'q3', q3_df, n_samples=n_samples)
                create_bootstrap_predictions(region, model_type, metric, 'q3', samples, n_trains)

def f(
    x: np.ndarray,
    params: Tuple[float]) -> Union[float, List[float]]:
    return -params[0] / (x - params[1]) + params[2]

def get_p_init(metric: str) -> Tuple[float, float, float]:
    if get_metric_direction(metric):
        return (1, -1, 1)
    else:
        return (-1, -1, 1)

# In which direction does the metric improve?
# Higher is better (True) or lower is better (False).
def get_metric_direction(metric: str) -> bool:
    if 'apl-mm-tol-' in metric:
        return False
    if metric == 'dice':
        return True
    if 'dm-surface-dice-tol-' in metric:
        return True
    if 'hd' in metric: 
        return False
    if metric == 'msd':
        return False

def fit_curve(
    p_init: Tuple[float],
    x: List[float],
    y: List[float], 
    max_nfev: int = DEFAULT_MAX_NFEV, 
    raise_error: bool = True, 
    weights: Optional[List[float]] = None):
    # Make fit.
    x_min = np.min(x)
    result = least_squares(__residuals(f), p_init, args=(x, y, weights), bounds=((-np.inf, -np.inf, 0), (np.inf, x_min, np.inf)), max_nfev=max_nfev)

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
    region: str,
    model: str,
    metric: str, 
    stat: str,
    include_params: bool = False,
    n_samples: int = DEFAULT_N_SAMPLES) -> Tuple[np.ndarray]:
    arg_assert_present(region, 'region')
    arg_assert_present(model, 'model')
    arg_assert_present(metric, 'metric')
    arg_assert_present(stat, 'stat')

    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'preds', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    if include_params:
        params = f['params']
        return data, params
    else:
        return data

def load_bootstrap_samples(
    region: str,
    model: str,
    metric: str,
    stat: str, 
    include_n_trains: bool = False,
    n_samples: int = DEFAULT_N_SAMPLES):
    dirpath = os.path.join(config.directories.files, 'transfer-learning', 'data', 'bootstrap', 'samples', model, metric, stat)
    filepath = os.path.join(dirpath, f'{region}-{n_samples}.npz')
    f = np.load(filepath)
    data = f['data']
    if include_n_trains:
        n_trains = f['n_trains']
        return data, n_trains
    else:
        return data
    
def load_bootstrap_differences(
    region: str,
    models: List[str],
    metric: str, 
    stat: str,
    n_samples: int = DEFAULT_N_SAMPLES) -> Tuple[List[int], Dict[int, str]]:
    assert len(models) == 2

    # Calculate fitted value differences.
    preds_a = load_bootstrap_predictions(region, models[0], metric, stat, n_samples=n_samples)
    preds_b = load_bootstrap_predictions(region, models[1], metric, stat, n_samples=n_samples)
    diffs = preds_b - preds_a

    # Calculate percentiles.
    diffs_p5 = np.quantile(diffs, 0.05, axis=0)
    diffs_p95 = np.quantile(diffs, 0.95, axis=0)

    # Build map from integer to model name.
    model_map = {
        0: 'No significant difference',
        1: models[0],
        2: models[1]
    }

    # If a model performs better in 95% of samples, it is significantly better
    # for that value of 'n_train'.
    results = np.zeros_like(diffs_p5, dtype=int)
    if get_metric_direction(metric):
        results[diffs_p5 > 0] = 2
        results[diffs_p95 < 0] = 1
    else:
        results[diffs_p95 < 0] = 2
        results[diffs_p5 > 0] = 1

    return results, model_map

def load_evaluation_data(
    datasets: Union[str, List[str]],
    regions: Union[str, List[str]],
    model_types: Union[str, List[str]],
    n_trains: Union[Optional[int], List[Optional[int]]] = DEFAULT_N_TRAINS,
    n_folds: Optional[int] = 5,
    test_folds: Union[int, List[int]] = list(range(5))) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = [datasets] if type(datasets) == str else datasets
    model_types = [model_types] if type(model_types) == str else model_types
    n_trains = [n_trains] if type(n_trains) == int else n_trains
    regions = [regions] if type(regions) == str else regions
    test_folds = [test_folds] if type(test_folds) == int else test_folds

    # Load evaluations and combine.
    dfs = []
    for region in regions:
        for test_fold in test_folds:
            # Add public evaluation.
            localiser = (f'localiser-{region}', 'public-1gpu-150epochs', 'BEST')
            seg_run = 'public-1gpu-150epochs'
            segmenter = (f'segmenter-{region}', seg_run, 'BEST')
            df = load_segmenter_evaluation(datasets, localiser, segmenter, n_folds=n_folds, test_fold=test_fold)
            df['model-type'] = seg_run
            df['n-train'] = 0
            dfs.append(df)

            # Get number of training cases.
            n_train_max = get_n_train_max(datasets, region, n_folds=n_folds, test_fold=test_fold)

            # Add clinical/transfer evaluations.
            for model in (model for model in model_types if model != 'public'):
                for n_train in n_trains:
                    # Skip if we've exceeded available number of training samples.
                    if n_train is not None and n_train > n_train_max:
                        continue

                    seg_run = f'{model}-fold-{test_fold}-samples-{n_train}'
                    segmenter = (f'segmenter-{region}', seg_run, 'BEST')
                    df = load_segmenter_evaluation(datasets, localiser, segmenter, test_fold=test_fold)
                    df['model-type'] = seg_run
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

    # Replace model names.
    df['model-type'] = df['model-type'].str.split('-').apply(lambda l: l[0])
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
    fontsize: float = DEFAULT_FONT_SIZE,
    legend_loc: Optional[Union[str, List[str]]] = None,
    model_label: Optional[Union[str, List[str]]] = None,
    savepath: Optional[str] = None,
    secondary_stat: Optional[Union[str, List[str], np.ndarray]] = None,
    y_lim: bool = True,
    **kwargs: Dict[str, Any]) -> None:
    datasets = arg_to_list(dataset, str)
    regions = arg_to_list(region, str)
    n_regions = len(regions)
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

    # Create main gridspec._labels
    fig = plt.figure(constrained_layout=False, figsize=(6 * metrics.shape[1], 6 * n_regions))
    gs = GridSpec(n_regions, metrics.shape[1], figure=fig)
    for i, region in enumerate(regions):
        for j in range(n_metrics):
            metric = metrics[i, j]
            stat = stats[j]
            sec_stat = secondary_stats[i, j]
            y_lim = DEFAULT_METRIC_Y_LIMS[metric] if y_lim else (None, None)
            legend_loc = legend_locs[j] if legend_locs is not None else DEFAULT_METRIC_LEGEND_LOCS[metric]
            plot_bootstrap_fit(datasets, region, models, metric, stat, fontsize=fontsize, subplot_spec=gs[i, j], legend_loc=legend_loc, model_labels=model_labels, secondary_stat=sec_stat, split=True, x_scale='log', y_label=DEFAULT_METRIC_LABELS[metric], y_lim=y_lim, **kwargs)

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_bootstrap_fit(
    datasets: Union[str, List[str]],
    region: str, 
    models: Union[str, List[str]],
    metric: str,
    stat: str,
    alpha_ci: float = 0.2,
    alpha_points: float = 1.0,
    alpha_secondary: float = 0.5,
    axs: Optional[Union[mpl.axes.Axes, List[mpl.axes.Axes]]] = None,
    diff_hspace: Optional[float] = 0.05,
    diff_marker: str = 's',
    figsize: Tuple[float, float] = (8, 6),
    fontsize: float = DEFAULT_FONT_SIZE,
    fontweight: Literal['normal', 'bold'] = 'normal',
    index: Optional[int] = None,
    legend_loc: Optional[str] = None,
    model_labels: Optional[Union[str, List[str]]] = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_trains: Optional[Union[Union[int, str], List[Union[int, str]]]] = DEFAULT_N_TRAINS,
    secondary_stat: Optional[str] = None,
    show_ci: bool = True,
    show_difference: bool = True,
    show_limits: bool = False,
    show_points: bool = True,
    show_secondary_ci: bool = True, 
    show_secondary_difference: bool = True,
    split: bool = True,
    split_wspace: Optional[float] = 0.05,
    subplot_spec: Optional[mpl.gridspec.SubplotSpec] = None,
    title: str = '',
    x_scale: str = 'log',
    y_label: str = '',
    y_lim: Optional[Tuple[float, float]] = None):
    model_colours = sns.color_palette('colorblind')[:len(models)]
    legend_loc = DEFAULT_METRIC_LEGEND_LOCS[metric] if legend_loc is None else legend_loc
    models = [models] if type(models) == str else models
    if model_labels is not None:
        model_labels = [model_labels] if type(model_labels) == str else model_labels
        assert len(model_labels) == len(models)
    if secondary_stat is None:
        show_secondary_difference = False
        
    # Create main gridspec.
    main_gs_size = (2, 1) if show_difference else (1, 1)
    main_gs_height_ratios = (48 if show_secondary_difference else 49, 2 if show_secondary_difference else 1) if show_difference else None
    if subplot_spec is None:
        fig = plt.figure(figsize=figsize)
        main_gs = GridSpec(*main_gs_size, figure=fig, height_ratios=main_gs_height_ratios)
    else:
        fig = subplot_spec.get_gridspec().figure
        main_gs = GridSpecFromSubplotSpec(*main_gs_size, subplot_spec=subplot_spec, height_ratios=main_gs_height_ratios)

    # Create data gridspec.
    data_gs_size = (1, 2) if split else (1, 1)
    data_gs_width_ratios = (1, 19) if split else None
    data_gs = GridSpecFromSubplotSpec(*data_gs_size, subplot_spec=main_gs[0, 0], width_ratios=data_gs_width_ratios, wspace=split_wspace)

    # Create difference gridspec.
    if show_difference:
        diff_gs_size = (2, 2) if show_secondary_difference else (1, 2)
        diff_gs_hspace = diff_hspace if show_secondary_difference else None
        diff_gs = GridSpecFromSubplotSpec(*diff_gs_size, hspace=diff_gs_hspace, subplot_spec=main_gs[1, 0], width_ratios=(1, 19), wspace=split_wspace)

    # Create subplots.
    axs = [
        [fig.add_subplot(data_gs[0, 0]), fig.add_subplot(data_gs[0, 1]) if split else None],
        [None, fig.add_subplot(diff_gs[0, 1]) if show_difference else None],
        [None, fig.add_subplot(diff_gs[1, 1]) if show_difference and show_secondary_difference else None]
    ]
    
    # Plot main data.
    for ax in axs[0]:
        if ax is None:
            continue

        for i, model in enumerate(models):
            model_colour = model_colours[i]
            model_label = model_labels[i] if model_labels is not None else model

            if index is not None:
                # Load bootstrapped sample.
                samples, n_trains = load_bootstrap_samples(region, model, metric, stat, include_n_trains=True, n_samples=n_samples)
                sample = samples[index]
                x_raw = np.array([])
                y_raw = np.array([])
                for n_train, n_train_sample in zip(n_trains, sample):
                    x_raw = np.concatenate((x_raw, n_train * np.ones(len(n_train_sample))))
                    y_raw = np.concatenate((y_raw, n_train_sample))

                # Load prediction on bootstrapped sample.
                preds = load_bootstrap_predictions(region, model, metric, stat, n_samples=n_samples)
                pred = preds[index]

                # Plot.
                x = np.linspace(0, len(pred) - 1, num=len(pred))
                ax.plot(x, pred, color=model_colour, label=model_label)
                if show_points:
                    ax.scatter(x_raw, y_raw, color=model_colour, marker='o', alpha=alpha_points)
            else:
                # Load bootstrapped predictions.
                preds = load_bootstrap_predictions(region, model, metric, stat, n_samples=n_samples)

                # Load data for secondary statistic.
                if secondary_stat:
                    sec_preds = load_bootstrap_predictions(region, model, metric, secondary_stat, n_samples=n_samples)

                # Plot mean value of 'stat' over all bootstrapped samples (convergent value).
                means = preds.mean(axis=0)
                x = np.linspace(0, len(means) - 1, num=len(means))
                ax.plot(x, means, color=model_colour, label=model_label)

                # Plot secondary statistic mean values.
                if secondary_stat:
                    sec_means = sec_preds.mean(axis=0)
                    ax.plot(x, sec_means, color=model_colour, alpha=alpha_secondary, linestyle='--')

                # Plot secondary statistic 95% CIs.
                if secondary_stat and show_secondary_ci:
                    low_ci = np.quantile(sec_preds, 0.025, axis=0)
                    high_ci = np.quantile(sec_preds, 0.975, axis=0)
                    ax.fill_between(x, low_ci, high_ci, color=model_colour, alpha=alpha_secondary * alpha_ci)

                # Plot 95% CIs for statistic.
                if show_ci:
                    low_ci = np.quantile(preds, 0.025, axis=0)
                    high_ci = np.quantile(preds, 0.975, axis=0)
                    ax.fill_between(x, low_ci, high_ci, color=model_colour, alpha=alpha_ci)

                # Plot upper/lower limits for statistic.
                if show_limits:
                    min = preds.min(axis=0)
                    max = preds.max(axis=0)
                    ax.plot(x, min, c='black', linestyle='--', alpha=0.5)
                    ax.plot(x, max, c='black', linestyle='--', alpha=0.5)

                # Plot original data (before bootstrapping was applied).
                if show_points:
                    x_raw, y_raw = __raw_data(datasets, region, model, metric, stat, n_trains=n_trains)
                    ax.scatter(x_raw, y_raw, color=model_colour, marker='o', alpha=alpha_points)

    # Plot difference.
    if show_difference:
        assert len(models) == 2

        # Load significant difference.
        diffs, _ = load_bootstrap_differences(region, models, metric, stat, n_samples=n_samples)
        
        # Plot model A points.
        diff_ax = axs[1][1] if split else axs[1][0]
        x = np.argwhere(diffs == 1).flatten()
        diff_ax.scatter(x, [0] * len(x), color=model_colours[0], marker=diff_marker)

        # Plot model B points.
        x = np.argwhere(diffs == 2).flatten()
        diff_ax.scatter(x, [0] * len(x), color=model_colours[1], marker=diff_marker)

        if show_secondary_difference:
            # Load significant difference.
            diffs, _ = load_bootstrap_differences(region, models, metric, secondary_stat, n_samples=n_samples)
            
            # Plot model A points.
            diff_ax = axs[2][1] if split else axs[2][0]
            x = np.argwhere(diffs == 1).flatten()
            diff_ax.scatter(x, [0] * len(x), color=model_colours[0], marker=diff_marker)

            # Plot model B points.
            x = np.argwhere(diffs == 2).flatten()
            diff_ax.scatter(x, [0] * len(x), color=model_colours[1], marker=diff_marker)

    # Set axis scale.
    if split:
        axs[0][1].set_xscale(x_scale)

        if show_difference:
            axs[1][1].set_xscale(x_scale)

            if show_secondary_difference:
                axs[2][1].set_xscale(x_scale)
    else:
        axs[0][0].set_xscale(x_scale)

        if show_difference:
            axs[1][0].set_xscale(x_scale)

            if show_secondary_difference:
                axs[2][0].set_xscale(x_scale)

    # Set x tick labels.
    x_upper_lim = None
    if split:
        axs[0][0].set_xticks([0])
        if show_difference:
            axs[1][1].get_xaxis().set_visible(False)
            axs[1][1].get_yaxis().set_visible(False)

            if show_secondary_difference:
                axs[2][1].get_xaxis().set_visible(False)
                axs[2][1].get_yaxis().set_visible(False)

        if x_scale == 'log':
            x_upper_lim = axs[0][1].get_xlim()[1]      # Record x upper lim as setting ticks overrides this.
            axs[0][1].set_xticks(LOG_SCALE_X_TICK_LABELS)
            axs[0][1].get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    else:
        if show_difference:
            axs[1][0].get_xaxis().set_visible(False)
            axs[1][0].get_yaxis().set_visible(False)

            if show_secondary_difference:
                axs[2][0].get_xaxis().set_visible(False)
                axs[2][0].get_yaxis().set_visible(False)

    # Set axis limits.
    if split:
        axs[0][0].set_xlim(-0.5, 0.5)
        axs[0][1].set_xlim(4.5, x_upper_lim)

        if show_difference:
            axs[1][1].set_xlim(4.5, x_upper_lim)

            if show_secondary_difference:
                axs[2][1].set_xlim(4.5, x_upper_lim)

    # Set y label.
    axs[0][0].set_ylabel(y_label, fontsize=fontsize, weight=fontweight)

    # Set y limits.
    axs[0][0].set_ylim(y_lim)
    if split:
        axs[0][1].set_ylim(y_lim)

    # Condense y ticks labels.
    y_ticks = axs[0][0].get_yticks()
    apply_formatting = False
    for y_tick in y_ticks:
        if y_tick >= 1000:
            apply_formatting = True
    if apply_formatting:
        def format_y_tick(y_tick: float):
            y_tick = int(y_tick / 1000)
            return f'{y_tick}k'
        axs[0][0].set_yticklabels([format_y_tick(y) for y in y_ticks])

    # Set y tick label fontsize.
    axs[0][0].tick_params(axis='x', which='major', labelsize=fontsize)
    axs[0][0].tick_params(axis='y', which='major', labelsize=fontsize)
    if split:
        axs[0][1].tick_params(axis='x', which='major', labelsize=fontsize)

    # Set title.
    title = title if title else region
    if split:
        axs[0][1].set_title(title, fontsize=fontsize, weight=fontweight)
    else:
        axs[0][0].set_title(title, fontsize=fontsize, weight=fontweight)

    # Add legend.
    if split:
        axs[0][1].legend(fontsize=fontsize, loc=legend_loc)
    else:
        axs[0][0].legend(fontsize=fontsize, loc=legend_loc)

    if split:
        # Hide axes' spines.
        axs[0][0].spines['right'].set_visible(False)
        axs[0][1].spines['left'].set_visible(False)
        axs[0][1].set_yticks([])

        # Add split between axes.
        d_x_0 = .1
        d_x_1 = .006
        d_y = .03
        kwargs = dict(transform=axs[0][0].transAxes, color='k', clip_on=False)
        axs[0][0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (-d_y / 2, d_y / 2), **kwargs)  # bottom-left diagonal
        axs[0][0].plot((1 - (d_x_0 / 2), 1 + (d_x_0 / 2)), (1 - (d_y / 2), 1 + (d_y / 2)), **kwargs)  # top-left diagonal
        kwargs = dict(transform=axs[0][1].transAxes, color='k', clip_on=False)
        axs[0][1].plot((-d_x_1 / 2, d_x_1 / 2), (-d_y / 2, d_y / 2), **kwargs)  # bottom-left diagonal
        axs[0][1].plot((-d_x_1 / 2, d_x_1 / 2), (1 - (d_y / 2), 1 + (d_y / 2)), **kwargs)  # top-left diagonal

def __raw_data(
    datasets: Union[str, List[str]],
    region: str,
    model_type: str,
    metric: str,
    stat: str,
    n_trains: Optional[Union[Union[int, str], List[Union[int, str]]]] = DEFAULT_N_TRAINS) -> np.ndarray:
    # Load evaluation data.
    df = load_evaluation_data(datasets, region, model_type, n_trains=n_trains)
    if stat == 'mean':
        stat_df = get_mean_evaluation_data(df)
    elif stat == 'q1':
        stat_df = get_q1_evaluation_data(df)
    elif stat == 'q3':
        stat_df = get_q3_evaluation_data(df)
    else:
        raise ValueError(f"Stat '{stat} not recognised.")

    # Get data points.
    data_df = stat_df[(stat_df.metric == metric) & (stat_df['model-type'] == model_type) & (stat_df.region == region)]
    x = data_df['n-train']
    y = data_df['value']
    return x, y

def __residuals(f):
    def inner(params, x, y, weights):
        y_pred = f(x, params)
        rs = y - y_pred
        if weights is not None:
            rs *= weights
        return rs
    return inner

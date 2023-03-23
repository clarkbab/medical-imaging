import numpy as np
from typing import Any, Dict, List, Literal, Tuple, Union

from mymi.metrics import higher_is_better
from mymi.utils import arg_assert_literal, arg_to_list

from .predictions import load_bootstrap_point_predictions, load_bootstrap_predictions

DEFAULT_N_SAMPLES = int(1e4)

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

    # Shuffle preds.
    np.random.shuffle(preds_a)
    np.random.shuffle(preds_b)
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

    # Shuffle preds.
    np.random.shuffle(preds_a)
    np.random.shuffle(preds_b)
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
    preds_a, _ = load_bootstrap_predictions(datasets, region, model_type_a, metric, stat, n_samples=n_samples)
    preds_b, _ = load_bootstrap_predictions(datasets, region, model_type_b, metric, stat, n_samples=n_samples)

    # Shuffle bootstrapped predictions.
    np.random.shuffle(preds_a)
    np.random.shuffle(preds_b)

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

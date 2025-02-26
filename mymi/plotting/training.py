from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
from typing import *

from mymi.models import load_training_metrics, layer_summary
from mymi.typing import *
from mymi.utils import *

from .plotting import sanitise_label

GRADIENT_METRICS = [
    'gradient-output-mean',
    'gradient-output-std',
    'gradient-weight-mean',
    'gradient-weight-std',
    'gradient-bias-mean',
    'gradient-bias-std',
]
OUTPUT_METRICS = [
    'output-mean',
    'output-std',
]
PARAMETER_METRICS = [
    'parameter-weight-mean',
    'parameter-weight-std',
    'parameter-bias-mean',
    'parameter-bias-std',
]

ALL_METRICS =  GRADIENT_METRICS + OUTPUT_METRICS + PARAMETER_METRICS

MODES = ('train', 'validate')

def plot_training_metrics(
    model: ModelName,
    bbox_to_anchor: Tuple[float, float] = (1.05, 1.00),
    exclude_modules: Union[str, List[str]] = [],
    label_length: int = 50,
    metrics: Union[str, List[str]] = 'default',
    mode: Literal['train', 'validate'] = 'train',
    modules: Union[str, List[str]] = 'all',
    modules_like: Union[str, List[str]] = [],       # Select modules based on regular expressions.
    n_legend_rows: int = 20,
    show_epochs: bool = True,
    show_legend: bool = True,
    show_loss: bool = True,
    step_interval: int = 50,
    step_lim: Tuple[Optional[int], Optional[int]] = (None, None)) -> None:
    metrics = arg_to_list(metrics, str, literals={
        'all': ALL_METRICS,
        'gradients': GRADIENT_METRICS,
        'outputs': OUTPUT_METRICS,
        'parameters': PARAMETER_METRICS,
    })

    # Load data and module names.
    df = load_training_metrics(model)
    df = df[df['mode'] == mode]
    all_modules = df['module'].dropna().unique().tolist()
    param_modules = df[df['metric'].isin(PARAMETER_METRICS)]['module'].dropna().unique().tolist()

    # Filter modules based on arguments.
    modules = arg_to_list(modules, str, literals={ 'all': all_modules })
    modules_like = arg_to_list(modules_like, str)
    def filter_fn(
        module: str,
        exclude_modules: List[str] = [],
        modules: List[str] = [],
        modules_like: List[str] = []) -> bool:
        # Check excluded modules.
        if module in exclude_modules:
            return False

        # Check literal modules.
        if module in modules:
            return True

        # Check regular expressions.
        for p in modules_like:
            if re.match(p, module) is not None:
                return True

        return False
    modules = list(filter(partial(filter_fn, exclude_modules=exclude_modules, modules=modules, modules_like=modules_like), all_modules))
    param_modules = [m for m in modules if m in param_modules]

    # Get legend info.
    legend_n_cols = int(np.ceil(len(modules) / n_legend_rows))
    legend_n_cols_param = int(np.ceil(len(param_modules) / n_legend_rows))

    # Add step groups for aggregation.
    # Recorded steps might be [0, 10, 20, 30, ...] but we only want to show
    # every 2nd recorded step and aggregate (average) the other values.
    steps = [s for i, s in enumerate(df['step'].unique()) if i % step_interval == 0]
    step_groups = pd.cut(df['step'], steps, labels=False, right=False)      # Use [a, b), not (a, b].
    df.insert(2, 'step-group', step_groups)
    
    # Due to aggregation, we don't have any values for step=0.
    steps.pop(0)
    
    # Get number of rows.
    n_rows = len(metrics)
    if show_loss:
        n_rows += 1     # First row is train/validate-loss.

    _, axs = plt.subplots(n_rows, 1, figsize=(32, 6 * len(metrics)), gridspec_kw={ 'hspace': 0.3 }, sharex=False, squeeze=False)
    axs = [a[0] for a in axs] # Remove columns.

    def plot_metric(
        mode: str,
        metric: str,
        ax: mpl.axes.Axes,
        label: Optional[str] = None,
        module: Optional[str] = None,
        step_lim: Tuple[Optional[int], Optional[int]] = (None, None)) -> None:
        # Plot data.
        mdf = df[(df['mode'] == mode) & (df['metric'] == metric)]
        if module is not None:
            mdf = mdf[mdf['module'] == module]
        agg_df = mdf.groupby('step-group').agg({ 'step': 'first', 'value': 'mean' })
        metric_steps, metric_values = agg_df['step'].tolist(), agg_df['value'].tolist()
        ax.plot(metric_steps, metric_values, label=label)

        # Configure axes.
        ax.set_xlabel('Step')
        ax.set_ylabel(m)
        step_lim = list(step_lim)
        if step_lim[0] is None:
            step_lim[0] = df['step'].min()
        if step_lim[1] is None:
            step_lim[1] = df['step'].max()
        ax.set_xlim(*step_lim)

        if show_epochs:
            epochs = df['epoch'].unique().tolist()
            # Get first step of each epoch.
            first_steps = df[['epoch', 'step']].drop_duplicates().groupby('epoch').first()['step'].tolist()

            # Limit the number of epochs displayed.
            max_epochs = 10
            if len(epochs) > max_epochs:
                f = int(np.ceil((len(epochs) / max_epochs)))
                epochs = [e for i, e in enumerate(epochs) if i % f == 0]
                first_steps = [s for i, s in enumerate(first_steps) if i % f == 0]

            epoch_ax = ax.twiny()
            epoch_ax.set_xticks(first_steps)
            epoch_ax.set_xticklabels(epochs)
            epoch_ax.set_xlabel('Epochs')
            epoch_ax.set_xlim(ax.get_xlim())

        if show_legend:
            ax.legend(bbox_to_anchor=bbox_to_anchor, ncol=ncol)

    # Plot losses.
    if show_loss:
        for m in MODES:
            plot_metric(m, 'loss', axs[0], label=f'{m}-loss', step_lim=step_lim)

    # Plot other metrics.
    for i, m in enumerate(metrics):
        row = i + 1 if show_loss else i 
        # Plot other metrics.
        ncol = legend_n_cols if m not in PARAMETER_METRICS else legend_n_cols_param

        # For 'parameter' metrics (e.g. 'parameter-bias-mean') exclude modules without parameters.
        metric_modules = modules if m not in PARAMETER_METRICS else param_modules
        for mod in metric_modules:
            plot_metric(mode, m, axs[row], label=sanitise_label(mod, max_length=label_length), module=mod, step_lim=step_lim)

    plt.show()

def plot_model_metrics(
    model: ModelName,
    arch: str,
    steps: List[int],
    *args,
    metrics: Union[str, List[str]] = 'all') -> None:
    metrics = arg_to_list(metrics, str, literals={
        'all': ALL_LAYER_METRICS,
        'activations': ACTIVATION_METRICS,
        'gradients': GRADIENT_METRICS,
        'parameters': PARAMETER_METRICS,
    })

    # Get modules.
    sum_df = layer_summary(arch, *args)
    x = list(sum_df['module'])

    # Get training metrics.
    df = load_training_metrics(model)
    # I think this is the validation metrics for different validation batches (same epoch/step).
    # # For some reason the same (or close) values are being added multiple times.
    # df['value'] = df['value'].round(6)
    # df = df.drop_duplicates()

    # Get modules.
    sum_df = layer_summary(arch, *args)
    x = list(sum_df['module'])

    _, axes = plt.subplots(len(metrics), len(steps), figsize=(len(steps) * 16, len(metrics) * 6), sharex=True, squeeze=False)
    for r in range(len(metrics)):
        for s in range(1, len(steps)):
            axes[r, s].sharey(axes[r, 0])   # Share y-axis across rows.

    for axs, m in zip(axes, metrics):
        for ax, s in zip(axs, steps):
            mdf = df[(df['metric'] == m) & (df['step'] == s)]
            modules, values = mdf['module'].tolist(), mdf['value'].tolist()
            ax.bar(modules, values)

            ax.tick_params(axis='x', labelrotation=90)
            ax.set_xlabel('Module')
            ax.set_ylabel(f'{m} (step={s})')

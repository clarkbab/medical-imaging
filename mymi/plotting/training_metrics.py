from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns
from typing import *

from mymi.reporting import load_training_metrics
from mymi.typing import *
from mymi.utils import *

from .plotting import sanitise_label

GRADIENT_METRICS = [
    # 'gradient-output-mean',
    # 'gradient-output-std',
    'gradient-output-l2',
    # 'gradient-weight-mean',
    # 'gradient-weight-std',
    # 'gradient-bias-mean',
    # 'gradient-bias-std',
    'gradient-weight-l2',
    'gradient-bias-l2',
]
OUTPUT_METRICS = [
    'output-mean',
    'output-std',
]
PARAMETER_METRICS = [
    'weight-mean',
    'weight-std',
    'bias-mean',
    'bias-std',
]

ALL_METRICS = PARAMETER_METRICS + OUTPUT_METRICS + GRADIENT_METRICS

DEFAULT_FONTSIZE = 16
MODES = ('train', 'validate')

def plot_training_metrics(
    model: ModelName,
    bbox_to_anchor: Tuple[float, float] = (1.05, 1.00),
    colourmap: str = 'plasma_r',
    exclude_modules: Union[str, List[str]] = [],
    fontsize: float = DEFAULT_FONTSIZE,
    label_length: int = 50,
    metrics: Union[str, List[str], List[List[str]]] = 'all',
    mode: Literal['train', 'validate'] = 'train',   # Applies to all metrics except for loss.
    modules: Union[str, List[str]] = 'all',
    modules_like: Union[str, List[str]] = [],       # Select modules based on regular expressions.
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
    df = df[(df['mode'] == mode) | (df['metric'] == 'loss')]  # Don't filter validation loss.
    all_modules = df['module'].dropna().unique().tolist()
    param_modules = df[df['metric'].isin(PARAMETER_METRICS)]['module'].dropna().unique().tolist()

    # Filter modules based on arguments.
    exclude_modules = arg_to_list(exclude_modules, str) 
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

    # Add step groups for aggregation.
    # Recorded steps might be [0, 10, 20, 30, ...] but we only want to show
    # every 2nd recorded step and aggregate (average) the other values.
    step_groups = np.arange(df['step'].min(), df['step'].max() + step_interval, step_interval).tolist()
    df['step-group'] = pd.cut(df['step'], step_groups, labels=False, right=False)      # Use [a, b), not (a, b].
    
    # Get number of rows.
    n_rows = len(metrics)
    if show_loss:
        n_rows += 1     # First row is train/validate-loss.

    _, axs = plt.subplots(n_rows, 1, figsize=(32, 8 * n_rows), gridspec_kw={ 'hspace': 0.3 }, sharex=False, squeeze=False)
    axs = [a[0] for a in axs] # Remove columns.

    def plot_metric(
        mode: str,
        metric: str,
        ax: mpl.axes.Axes,
        colour: Optional[Colour] = None,
        label: Optional[str] = None,
        module: Optional[str] = None,) -> None:
        # Plot data.
        mdf = df[(df['mode'] == mode) & (df['metric'] == metric)]
        if module is not None:
            mdf = mdf[mdf['module'] == module]
        agg_df = mdf.groupby('step-group').agg({ 'step': 'last', 'value': 'mean' })
        metric_steps, metric_values = agg_df['step'].tolist(), agg_df['value'].tolist()
        if len(metric_steps) != 0:
            ax.plot(metric_steps, metric_values, color=colour, label=label)

    def configure_axes(
        ax: mpl.axes.Axes,
        y_label: str,
        fontsize: float = DEFAULT_FONTSIZE,
        legend_n_cols: int = 1,
        show_legend: bool = True,
        step_lim: Tuple[Optional[int], Optional[int]] = (None, None)) -> None:
        # Configure axes.
        ax.grid()
        ax.set_xlabel('Step', fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)
        step_lim = list(step_lim)
        if step_lim[0] is None:
            step_lim[0] = df['step'].min()
        if step_lim[1] is None:
            step_lim[1] = df['step'].max()
        ax.set_xlim(*step_lim)
        ax.tick_params(axis='both', labelsize=fontsize)
        
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
            epoch_ax.tick_params(axis='x', labelsize=fontsize)
            epoch_ax.set_xlabel('Epochs', fontsize=fontsize)
            epoch_ax.set_xlim(ax.get_xlim())

        if show_legend:
            legend = ax.legend(bbox_to_anchor=bbox_to_anchor, fontsize=fontsize, ncol=legend_n_cols)
            for l in legend.get_lines():
                l.set_linewidth(8)

    # Plot losses.
    if show_loss:
        for m in MODES:
            plot_metric(m, 'loss', axs[0], label=f'{m}-loss')
        configure_axes(axs[0], 'loss', fontsize=fontsize, show_legend=show_legend, step_lim=step_lim)

    # Plot other metrics.
    non_module_metrics = df[df['module'].isna()]['metric'].unique()
    for i, m in enumerate(metrics):
        ms = arg_to_list(m, str)
        show_metric_label = True if len(ms) > 1 else False
        # There might be more colours than we need, because 'metric_modules' might be smaller
        # than modules. But this is the only way to handle multiple metrics, each of which
        # might have a different number of modules.
        colours = sns.color_palette(colourmap, len(modules) * len(ms))

        # Plot multiple metrics on the same plot.
        show_module_label = False
        for j, mj in enumerate(ms):
            # For 'parameter' metrics (e.g. 'parameter-bias-mean') exclude modules without parameters.
            metric_modules = modules if mj not in PARAMETER_METRICS else param_modules
            # Some metrics aren't associated with modules, (e.g. 'validate-dice').
            if mj in non_module_metrics:
                metric_modules = [None]
            if len(metric_modules) > 1:
                show_module_label = True
            row = i + 1 if show_loss else i 
            for k, mod in enumerate(metric_modules):
                colour = colours[j * len(metric_modules) + k]
                if show_metric_label and show_module_label:
                    y_label = ''
                    label = f'{label}:{mj}'
                elif show_metric_label:
                    y_label = mod
                    label = mj
                elif show_module_label:
                    y_label = mj
                    label = mod
                else:
                    y_label = mj
                    label = mod
                plot_metric(mode, mj, axs[row], colour=colour, label=sanitise_label(label, max_length=label_length), module=mod)

        if show_metric_label and show_module_label:
            y_label = ''
        elif show_metric_label:
            y_label = mod
        elif show_module_label:
            y_label = mj
        else:
            y_label = mj
        configure_axes(axs[row], y_label, fontsize=fontsize, show_legend=show_legend, step_lim=step_lim)

    plt.show()

def plot_model_metrics(
    model: ModelName,
    steps: List[int],
    exclude_modules: Union[str, List[str]] = [],
    fontsize: float = DEFAULT_FONTSIZE,
    metrics: Union[str, List[str]] = 'all',
    mode: str = 'train',
    modules: Union[str, List[str]] = 'all',
    modules_like: Union[str, List[str]] = [],
    show_diff: bool = True) -> None:
    metrics = arg_to_list(metrics, str, literals={
        'all': ALL_METRICS,
        'gradients': GRADIENT_METRICS,
        'outputs': OUTPUT_METRICS,
        'parameters': PARAMETER_METRICS,
    })

    # Get training metrics.
    df = load_training_metrics(model)
    df = df[df['mode'] == mode]

    # Filter modules based on arguments.
    exclude_modules = arg_to_list(exclude_modules, str) 
    all_modules = df['module'].dropna().unique().tolist()
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
    df = df[df['module'].isin(modules)]

    n_cols = len(steps)
    if len(steps) == 1:
        show_diff = False
    if show_diff:
        n_cols += 1
    _, axes = plt.subplots(len(metrics), n_cols, figsize=(n_cols * 16, len(metrics) * 6), sharex=True, squeeze=False)

    # Share y-axes across metrics, including diff.
    for r in range(len(metrics)):
        for s in range(1, n_cols):
            axes[r, s].sharey(axes[r, 0])

    # Plot metrics.
    for axs, m in zip(axes, metrics):
        # Skip last column, this will be difference.
        if show_diff:
            axs = axs[:-1]

        for ax, s in zip(axs, steps):
            mdf = df[(df['metric'] == m) & (df['step'] == s)]
            mods, values = mdf['module'].tolist(), mdf['value'].tolist()
            ax.bar(mods, values)

            ax.grid(axis='y', linestyle='--')
            ax.tick_params(axis='x', labelrotation=90, labelsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            ax.set_xlabel('Module', fontsize=fontsize)
            ax.set_ylabel(f'{m} (step={s})', fontsize=fontsize)

    # Plot differences.
    if show_diff:
        step_1, step_2 = steps[0], steps[-1]
        for axs, m in zip(axes, metrics):
            step_1_df = df[(df['metric'] == m) & (df['step'] == step_1)].reset_index(drop=True)
            step_2_df = df[(df['metric'] == m) & (df['step'] == step_2)].reset_index(drop=True)
            mods, diffs = step_2_df['module'].tolist(), (step_2_df['value'] - step_1_df['value']).tolist()
            axs[-1].bar(mods, diffs, color='red')

            axs[-1].grid(axis='y', linestyle='--')
            axs[-1].tick_params(axis='x', labelrotation=90, labelsize=fontsize)
            axs[-1].tick_params(axis='y', labelsize=fontsize)
            axs[-1].set_xlabel('Module', fontsize=fontsize)
            axs[-1].set_ylabel(f'diff (steps={step_2},{step_1})', fontsize=fontsize)

    plt.show()

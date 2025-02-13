import matplotlib.pyplot as plt
import pandas as pd
from typing import *

from mymi.typing import *
from mymi.utils import *

ALL_METRICS = [
    'activation-min',
    'activation-max',
    'activation-mean',
    'activation-std',
    'parameter-min',
    'parameter-max',
    'parameter-mean',
    'parameter-std',
    'gradient-min',
    'gradient-max',
    'gradient-mean',
    'gradient-std',
]

def plot_layer_stats(
    model: ModelName,
    metrics: Union[str, List[str]] = 'all',
    modules: Union[str, List[str]] = 'all',
    show_epochs: bool = True) -> None:
    metrics = arg_to_list(metrics, str, literals={ 'all': ALL_METRICS })

    # Load data.
    filepath = os.path.join(config.directories.models, model[0], model[1], 'activation-stats.csv')
    act_df = load_csv(filepath)
    filepath = os.path.join(config.directories.models, model[0], model[1], 'parameter-stats.csv')
    param_df = load_csv(filepath)
    df = pd.concat([act_df, param_df], axis=0)
    modules = arg_to_list(modules, str, literals={ 'all': df['module'].unique().tolist() })

    _, axs = plt.subplots(len(metrics), 1, figsize=(16, 6 * len(metrics)), gridspec_kw={ 'hspace': 0.3 }, sharex=False)

    # Plot each metric.
    for ax, m in zip(axs, metrics):
        for mod in modules:
            mod_df = df[df['module'] == mod]
            ax.plot(mod_df['step'], mod_df[m], label=mod[1:])
        ax.set_xlabel('Step')
        ax.set_ylabel(m)
        ax.set_xlim(df['step'].min(), df['step'].max())

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

        # ax.legend()
    plt.show()

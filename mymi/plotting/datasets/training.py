from typing import *

from mymi.datasets import TrainingDataset
from mymi.typing import *
from mymi.utils import *

from ..plotting import *

def plot_dataset_histogram(
    datasets: Union[str, List[str]],
    n_samples: Optional[int] = None,
    sample_ids: Optional[PatientIDs] = [],
    split_id: Optional[Split] = None,
    **kwargs) -> None:
    datasets = arg_to_list(datasets, str)
    _, axs = plt.subplots(1, len(datasets), figsize=(12 * len(datasets), 6), sharex=True, squeeze=False)

    for d, ax in zip(datasets, axs[0]):
        set = TrainingDataset(d)
        if split_id is None:
            split = set.split(set.list_splits()[0])
        else:
            split = set.split(split_id)

        # Filter on sample IDs.
        sids = sample_ids
        if n_samples is not None:
            sids = split.list_samples()
            sids = sids[:n_samples]

        inputs = [split.sample(s).input for s in sids]
        inputs = np.concatenate([i.flatten() for i in inputs])

        plot_histogram(inputs, ax=ax, title=d, **kwargs)

    plt.show()

def plot_samples(
    dataset: str,
    split_ids: Splits = 'all',
    sample_ids: SampleIDs = 'all',
    centre: Optional[str] = None,
    channels: Union[int, List[int], Literal['all']] = 0,
    **kwargs) -> None:
    set = TrainingDataset(dataset)
    split_ids = arg_to_list(split_ids, str, literals={ 'all': set.splits })
    channels = arg_to_list(channels, int, literals={ 'all': list(range(set.n_input_channels)) })
    n_channels = len(channels)

    # Load CT data.
    plot_idses = []
    inputses = []
    spacingses = []
    centreses = []
    for s in split_ids:
        # Get split samples.
        split = set.split(s)
        split_sample_ids = arg_to_list(sample_ids, SampleID, literals={ 'all': split.list_samples })
        for ss in split_sample_ids:
            plot_ids = []
            inputs = []
            spacings = []
            centres = []

            for c in channels:
                # Add sample data.
                plot_id = f'{s}:{ss}:{c}'
                sample = split.sample(ss)
                input = sample.input[c]
                spacing = set.spacing

                plot_ids.append(plot_id)
                inputs.append(input)
                spacings.append(spacing)
                centres.append(centre)

            # Collapse list if single channel.
            if n_channels == 1:
                plot_ids = plot_ids[0]
                inputs = inputs[0]
                spacings = spacings[0]
                centres = centres[0]

            plot_idses.append(plot_ids)
            inputses.append(inputs)
            spacingses.append(spacings)
            centreses.append(centres)

    # Plot.
    okwargs = dict(
        centres=centreses,
        ct_datas=inputses,
    )
    plot_patients_matrix(plot_idses, spacingses, **okwargs, **kwargs)

def plot_sample_histograms(
    dataset: str,
    split_ids: Splits = 'all',
    sample_ids: PatientIDs = 'all',
    channels: Optional[Channels] = 'all',
    **kwargs) -> None:
    set = TrainingDataset(dataset)
    split_ids = arg_to_list(split_ids, str, literals={ 'all': set.splits })
    channels = arg_to_list(channels, int, literals={ 'all': list(range(set.n_input_channels)) })
    row_ids = []    # ('split_id', 'sample_id')
    for s in split_ids:
        split = set.split(s)
        split_sample_ids = arg_to_list(sample_ids, SampleID, literals={ 'all': split.list_samples })
        for ss in split_sample_ids:
            row_ids.append((s, ss))
    n_rows = len(row_ids)
    n_cols = len(channels) if channels is not None else 1
    
    _, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    for row_axs, (s, ss) in zip(axs, row_ids):
        sample = set.split(s).sample(ss)
        input = sample.input
        if channels is None:
            title = f"{s}:{ss}"
            plot_histogram(input.flatten(), ax=row_axs[0], title=title, **kwargs)
        else:
            for col_ax, c in zip(row_axs, channels):
                title = f"{s}:{ss}:{c}"
                plot_histogram(input[c].flatten(), ax=col_ax, title=title, **kwargs)
            
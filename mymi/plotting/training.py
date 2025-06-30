from typing import *

from mymi.datasets import TrainingDataset
from mymi.typing import *
from mymi.utils import *

from ..plotting import *

def plot_samples(
    dataset: str,
    split_ids: Splits = 'all',
    sample_ids: SampleIDs = 'all',
    centre: Optional[str] = None,
    channels: Union[int, List[int], Literal['all']] = 0,
    **kwargs) -> None:
    set = TrainingDataset(dataset)
    spacing = set.spacing
    split_ids = arg_to_list(split_ids, str, literals={ 'all': set.list_splits })
    channels = arg_to_list(channels, int, literals={ 'all': list(range(set.n_input_channels)) })

    # Load CT data.
    plot_idses = []
    inputses = []
    spacingses = []
    centreses = []
    series_idses = []
    for s in split_ids:
        # Get split samples.
        split = set.split(s)
        split_sample_ids = arg_to_list(sample_ids, SampleID, literals={ 'all': split.list_samples })
        for ss in split_sample_ids:
            plot_ids = []
            inputs = []
            spacings = []
            centres = []
            series_ids = []

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
                series_ids.append(c)

            plot_idses.append(plot_ids)
            inputses.append(inputs)
            spacingses.append(spacings)
            centreses.append(centres)
            series_idses.append(series_ids)

    # Plot.
    n_rows = len(inputses)
    okwargs = dict(
        centres=centreses,
        datas=inputses,
        landmark_datas=None,
        region_imagess=None,
        series_ids=series_idses,
        spacings=[spacing] * n_rows,
    )
    plot_patients_matrix(plot_idses, **okwargs, **kwargs)

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
            
from mymi.predictions.datasets.nifti import create_unigradicon_predictions

dataset = 'DIRLAB-LUNG-COPD-CP'
model = 'unigradicon'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    use_io=False,
)
create_unigradicon_predictions(dataset, model, **kwargs)

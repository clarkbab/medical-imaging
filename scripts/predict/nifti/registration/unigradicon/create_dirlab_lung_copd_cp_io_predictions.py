from mymi.predictions.datasets.nifti import create_unigradicon_predictions

dataset = 'DIRLAB-LUNG-COPD-CP'
model = 'unigradicon-io'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    use_io=True,
)
create_unigradicon_predictions(dataset, model, **kwargs)

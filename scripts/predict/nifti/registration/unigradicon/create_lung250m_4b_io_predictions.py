from mymi.predictions.datasets.nifti import create_unigradicon_predictions

dataset = 'LUNG250M-4B'
model = 'unigradicon-io'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    splits='test',
    use_io=True,
)
create_unigradicon_predictions(dataset, model, **kwargs)

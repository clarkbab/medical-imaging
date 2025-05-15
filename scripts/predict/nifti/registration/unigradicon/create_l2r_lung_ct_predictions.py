from mymi.predictions.datasets.nifti import create_unigradicon_predictions

dataset = 'L2R-LUNG-CT'
model = 'unigradicon'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    splits='test',
    use_io=False,
)
create_unigradicon_predictions(dataset, model, **kwargs)

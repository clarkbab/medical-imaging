from mymi.predictions.nifti import create_unigradicon_predictions

dataset = 'L2R-LUNG-CT'
model = 'unigradicon-io'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    splits='test',
    use_io=True,
)
create_unigradicon_predictions(dataset, model, **kwargs)

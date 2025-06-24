from mymi.predictions.nifti import create_unigradicon_predictions

dataset = 'LUNG250M-4B'
model = 'unigradicon'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    splits='test',
    use_io=False,
)
create_unigradicon_predictions(dataset, model, **kwargs)

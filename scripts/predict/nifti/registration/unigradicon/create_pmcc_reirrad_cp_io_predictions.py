from mymi.predictions.nifti import create_unigradicon_predictions

dataset = 'PMCC-REIRRAD-CP'
model = 'unigradicon-io'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    use_io=True,
)
create_unigradicon_predictions(dataset, model, **kwargs)

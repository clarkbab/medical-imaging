from mymi.predictions.nifti import create_unigradicon_predictions, warp_patients_data

dataset = 'PMCC-REIRRAD-CP'
model = 'unigradicon'
kwargs = dict(
    landmarks='all',
    register_ct=True,
    regions='all',
    use_io=False,
)
# create_unigradicon_predictions(dataset, model, **kwargs)
warp_patients_data(dataset, model, **kwargs)

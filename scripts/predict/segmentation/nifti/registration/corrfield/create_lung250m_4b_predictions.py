from mymi.predictions.nifti.registration import create_corrfield_predictions

dataset = 'LUNG250M-4B'
kwargs = dict(
    preprocess_images=False,
    splits='test',
)
create_corrfield_predictions(dataset, **kwargs)

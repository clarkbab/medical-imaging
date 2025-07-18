from mymi.predictions.nifti import create_deeds_predictions

dataset = 'LUNG250M-4B'
kwargs = dict(
    preprocess_images=False,
    splits='test',
)
create_deeds_predictions(dataset, **kwargs)

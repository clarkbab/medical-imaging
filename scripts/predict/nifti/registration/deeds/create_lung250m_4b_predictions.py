from mymi.predictions.datasets.nifti import create_deeds_predictions

dataset = 'LUNG250M-4B'
kwargs = dict(
    preprocess_images=False,
)
create_deeds_predictions(dataset, **kwargs)

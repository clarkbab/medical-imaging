from mymi.predictions.datasets.nifti.registration import create_corrfield_predictions

dataset = 'DIRLAB-LUNG-COPD-CP'
kwargs = dict(
    preprocess_images=False,
)
create_corrfield_predictions(dataset, **kwargs)

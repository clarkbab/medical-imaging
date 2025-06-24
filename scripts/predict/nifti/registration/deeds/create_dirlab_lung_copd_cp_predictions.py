from mymi.predictions.nifti import create_deeds_predictions

dataset = 'DIRLAB-LUNG-COPD-CP'
kwargs = dict(
    preprocess_images=False,
)
create_deeds_predictions(dataset, **kwargs)

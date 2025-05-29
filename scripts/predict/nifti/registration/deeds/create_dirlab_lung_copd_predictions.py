from mymi.predictions.datasets.nifti import create_deeds_predictions

dataset = 'DIRLAB-LUNG-COPD'
kwargs = dict(
    lung_region='Lung',
)
create_deeds_predictions(dataset, **kwargs)

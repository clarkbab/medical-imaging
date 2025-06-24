from mymi.predictions.nifti.registration import create_corrfield_predictions

dataset = 'DIRLAB-LUNG-COPD'
kwargs = dict(
    lung_region='Lung',
    # pat_ids='copd8',
)
create_corrfield_predictions(dataset, **kwargs)

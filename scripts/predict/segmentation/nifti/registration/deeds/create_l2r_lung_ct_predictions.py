from mymi.predictions.nifti import create_deeds_predictions

dataset = 'L2R-LUNG-CT'
kwargs = dict(
    lung_region='Lung',
    splits='test',
)
create_deeds_predictions(dataset, **kwargs)

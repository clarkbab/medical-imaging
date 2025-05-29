from mymi.predictions.datasets.nifti.registration import create_corrfield_predictions

dataset = 'L2R-LUNG-CT'
kwargs = dict(
    lung_region='Lung',
    splits='test',
)
create_corrfield_predictions(dataset, **kwargs)

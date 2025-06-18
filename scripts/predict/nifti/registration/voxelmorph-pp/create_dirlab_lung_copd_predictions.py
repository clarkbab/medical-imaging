from mymi.predictions.datasets.nifti import create_voxelmorph_pp_predictions

dataset = 'DIRLAB-LUNG-COPD'
kwargs = dict(
    crop_to_lung_centres=True,
    lung_region='Lung',
    perform_breath_resample=True,
)
create_voxelmorph_pp_predictions(dataset, **kwargs)
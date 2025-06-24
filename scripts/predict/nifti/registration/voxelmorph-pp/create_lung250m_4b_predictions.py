from mymi.predictions.nifti import create_voxelmorph_pp_predictions

dataset = 'LUNG250M-4B'
kwargs = dict(
    crop_to_lung_centres=True,
    perform_breath_resample=True,
    splits='test',
)
create_voxelmorph_pp_predictions(dataset, **kwargs)
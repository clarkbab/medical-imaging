from mymi.predictions.datasets.nifti import create_voxelmorph_pp_predictions

dataset = 'DIRLAB-LUNG-COPD-VXMPP'
kwargs = dict(
    # splits='test',
)
create_voxelmorph_pp_predictions(dataset, **kwargs)
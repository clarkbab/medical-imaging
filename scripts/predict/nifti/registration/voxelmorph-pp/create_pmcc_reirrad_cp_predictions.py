from mymi.predictions.datasets.nifti import create_voxelmorph_pp_predictions

dataset = 'PMCC-REIRRAD-CP'
kwargs = dict(
    # splits='test',
)
create_voxelmorph_pp_predictions(dataset, **kwargs)
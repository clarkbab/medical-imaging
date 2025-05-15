from mymi.predictions.datasets.nifti.registration import create_voxelmorph_predictions

dataset = 'L2R-LUNG-CT'
model = 'L2R-LUNG-CT-222-static/1500.pt'
model_name = 'L2R-LUNG-CT-222-static'
model_spacing = (2, 2, 2)
kwargs = dict(
    register_ct=True,
    landmarks='all',
    regions='all',
    splits='test',
)

create_voxelmorph_predictions(dataset, model, model_name, model_spacing, **kwargs)

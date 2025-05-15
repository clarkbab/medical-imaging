from mymi.predictions.datasets.nifti.registration import create_voxelmorph_predictions

dataset = 'LUNG250M-4B'
model = 'LUNG250M-4B-222/1500.pt'
model_name = 'LUNG250M-4B-222'
model_spacing = (2, 2, 2)
kwargs = dict(
    register_ct=True,
    landmarks='all',
    regions='all',
    splits='test',
)

create_voxelmorph_predictions(dataset, model, model_name, model_spacing, **kwargs)

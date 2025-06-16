from mymi.predictions.datasets.nifti.registration import create_voxelmorph_predictions

dataset = 'PMCC-REIRRAD-CP'
model = 'LUNG250M-4B-222-dynamic-2000'
full_model = f'{model}/1500.pt'
model_name = f'voxelmorph-{model}'
model_spacing = (2, 2, 2)
kwargs = dict(
    landmarks='all',
    # pat_ids='021',
    regions='all',
    # splits='test',
)

create_voxelmorph_predictions(dataset, full_model, model_name, model_spacing, **kwargs)

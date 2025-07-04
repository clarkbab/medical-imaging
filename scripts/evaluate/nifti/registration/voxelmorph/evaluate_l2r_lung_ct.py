from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'L2R-LUNG-CT'
model = 'voxelmorph-L2R-LUNG-CT-222-dynamic-2000'
kwargs = dict(
    splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

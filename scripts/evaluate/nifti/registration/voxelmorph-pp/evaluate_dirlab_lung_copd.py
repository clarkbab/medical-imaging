from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'DIRLAB-LUNG-COPD'
model = 'voxelmorph-pp'
kwargs = dict(
    # splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'PMCC-REIRRAD-CP'
model = 'voxelmorph-LUNG250M-4B-222-dynamic-2000'
kwargs = dict(
    # splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

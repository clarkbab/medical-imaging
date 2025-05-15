from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'L2R-LUNG-CT'
model ='identity'
kwargs = dict(
    splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

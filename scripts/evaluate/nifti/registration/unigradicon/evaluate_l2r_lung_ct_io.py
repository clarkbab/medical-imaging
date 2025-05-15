from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'L2R-LUNG-CT'
model = 'unigradicon-io'
kwargs = dict(
    splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

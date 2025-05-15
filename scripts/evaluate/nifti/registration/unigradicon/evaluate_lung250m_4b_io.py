from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'LUNG250M-4B'
model = 'unigradicon-io'
kwargs = dict(
    splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

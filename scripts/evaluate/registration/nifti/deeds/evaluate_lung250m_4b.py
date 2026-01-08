from mymi.evaluations.datasets.nifti import create_registration_evaluations

dataset = 'LUNG250M-4B'
model ='deeds'
kwargs = dict(
    splits='test',
)

create_registration_evaluations(dataset, model, **kwargs)

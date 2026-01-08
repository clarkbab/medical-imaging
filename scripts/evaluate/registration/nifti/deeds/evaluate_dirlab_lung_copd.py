from mymi.evaluations.datasets.nifti import create_registration_evaluations

dataset = 'DIRLAB-LUNG-COPD'
model ='deeds'
kwargs = dict(
    # splits='test',
)

create_registration_evaluations(dataset, model, **kwargs)

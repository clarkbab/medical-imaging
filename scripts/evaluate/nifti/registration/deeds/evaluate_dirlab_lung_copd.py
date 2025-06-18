from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'DIRLAB-LUNG-COPD'
model ='deeds'
kwargs = dict(
    # splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

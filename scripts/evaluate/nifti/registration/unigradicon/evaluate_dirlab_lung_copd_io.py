from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'DIRLAB-LUNG-COPD'
model = 'unigradicon-io'

create_registrations_evaluation(dataset, model)

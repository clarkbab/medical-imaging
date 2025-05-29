from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'DIRLAB-LUNG-COPD-CP'
model = 'unigradicon'

create_registrations_evaluation(dataset, model)

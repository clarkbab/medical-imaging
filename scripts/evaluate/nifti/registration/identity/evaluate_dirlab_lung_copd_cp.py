from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'DIRLAB-LUNG-COPD-CP'
model ='identity'

create_registrations_evaluation(dataset, model)

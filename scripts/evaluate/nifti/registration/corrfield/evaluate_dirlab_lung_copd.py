from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'DIRLAB-LUNG-COPD'
model ='corrfield'

create_registrations_evaluation(dataset, model)

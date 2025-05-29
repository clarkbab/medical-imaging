from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'LUNG-PROC'
model ='corrfield'

create_registrations_evaluation(dataset, model)

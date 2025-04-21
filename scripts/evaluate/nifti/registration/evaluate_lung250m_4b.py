from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'LUNG250M-4B'
model = 'lung250m-lambda=0.02'

create_registrations_evaluation(dataset, model)

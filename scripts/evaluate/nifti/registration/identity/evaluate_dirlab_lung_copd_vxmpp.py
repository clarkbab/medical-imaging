from mymi.evaluations.datasets.nifti import create_registrations_evaluation

dataset = 'DIRLAB-LUNG-COPD-VXMPP'
model ='vxmpp-identity'

create_registrations_evaluation(dataset, model)

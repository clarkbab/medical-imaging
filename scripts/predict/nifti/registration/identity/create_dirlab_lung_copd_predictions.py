from mymi.predictions.datasets.nifti import create_registrations

dataset = "DIRLAB-LUNG-COPD"
project = "IMREG"
model = 'identity'
model_spacing = None

create_registrations(dataset, project, model, model_spacing)

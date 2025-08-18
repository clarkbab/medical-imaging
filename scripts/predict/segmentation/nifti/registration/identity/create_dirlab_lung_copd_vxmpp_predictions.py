from mymi.predictions.nifti import create_registrations

dataset = "DIRLAB-LUNG-COPD-VXMPP"
project = "IMREG"
model = 'vxmpp-identity'
model_spacing = None

create_registrations(dataset, project, model, model_spacing)

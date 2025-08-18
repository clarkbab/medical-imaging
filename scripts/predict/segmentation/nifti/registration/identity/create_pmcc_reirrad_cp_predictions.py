from mymi.predictions.nifti import create_registrations, warp_patients_data

dataset = "PMCC-REIRRAD-CP"
project = "IMREG"
model = 'identity'
model_spacing = None

# create_registrations(dataset, project, model, model_spacing)
warp_patients_data(dataset, model)

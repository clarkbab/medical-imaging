from mymi.predictions.datasets.nifti import create_registrations

dataset = "LUNG250M-4B"
project = "IMREG"
model = 'identity'
model_spacing = None
kwargs = dict(
    splits='test',
)

create_registrations(dataset, project, model, model_spacing, **kwargs)

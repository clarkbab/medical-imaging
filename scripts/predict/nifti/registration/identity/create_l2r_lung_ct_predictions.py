from mymi.predictions.datasets.nifti import create_dataset_registrations

dataset = "L2R-LUNG-CT"
project = "IMREG"
model = 'identity'
model_spacing = None
kwargs = dict(
    splits='test',
)

create_dataset_registrations(dataset, project, model, model_spacing, **kwargs)

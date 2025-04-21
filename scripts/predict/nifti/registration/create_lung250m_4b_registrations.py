from mymi.predictions.datasets.nifti import create_dataset_registrations

dataset = "LUNG250M-4B"
project = "IMREG"
model = "lung250m-lambda=0.02"
model_spacing = (2, 2, 2)
kwargs = dict(
    splits='test',
)

create_dataset_registrations(dataset, project, model, model_spacing, **kwargs)

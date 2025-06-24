from mymi.predictions.nifti import create_registrations
from mymi.utils import grid_arg

loss_lambda = grid_arg('loss_lambda', 0.02)

dataset = "LUNG250M-4B"
project = "IMREG"
model = f"lung250m-222-lambda={loss_lambda}"
model_spacing = (2, 2, 2)
kwargs = dict(
    splits='test',
)

create_registrations(dataset, project, model, model_spacing, **kwargs)

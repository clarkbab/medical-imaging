from mymi.evaluations.datasets.nifti import create_registrations_evaluation
from mymi.utils import grid_arg

loss_lambda = grid_arg('loss_lambda', 0.02)

dataset = 'LUNG250M-4B'
model = f'lung250m-222-lambda={loss_lambda}'
kwargs = dict(
    splits='test',
)

create_registrations_evaluation(dataset, model, **kwargs)

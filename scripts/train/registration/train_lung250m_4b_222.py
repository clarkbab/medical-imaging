from mymi.training import train_registration
from mymi.utils import parse_arg

# Overridden args.
loss_lambda = parse_arg('loss_lambda', float, 0.02)
 
dataset = "LUNG250M-4B-222"
project = "IMREG"
lr_init = 1e-4
n_epochs = 5000
kwargs = dict(
    loss_fn="mse",
    loss_lambda=loss_lambda,
    # Padding voxels (at -1024) from the moving image were being matched to lung in the fixed image.
    # Discourage this by setting padding values well below air.
    # Change later as we're running alpha grid search.
    pad_fill=-1024,     
    resume=False,
)
model = f"lung250m-222-lambda={loss_lambda}"

train_registration(dataset, project, model, n_epochs, lr_init, **kwargs)

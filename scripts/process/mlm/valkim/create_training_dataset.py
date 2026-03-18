from mymi.processing.nifti import create_valkim_training_dataset

kwargs = dict(
    create_train_volumes=False,
    create_val_volumes=False,
    create_val_projections=True,
    makeitso=True,
    n_val_angles=100,
    n_val_volumes=10,
)
create_valkim_training_dataset(**kwargs)

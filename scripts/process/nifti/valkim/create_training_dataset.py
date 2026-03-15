from mymi.processing.nifti import create_valkim_training_dataset

kwargs = dict(
    create_train_volumes=False,
    create_val_volumes=True,
    create_val_projections=True,
    makeitso=True,
)
create_valkim_training_dataset(**kwargs)

from mymi.processing.datasets.nifti import create_registered_dataset

dataset = 'LUNG250M-4B'
dest_dataset = 'LUNG250M-4B-REG-NOXY'
kwargs = dict(
    disable_x_axis_rotation=True,
    disable_y_axis_rotation=True,
)
create_registered_dataset(dataset, dest_dataset, **kwargs)
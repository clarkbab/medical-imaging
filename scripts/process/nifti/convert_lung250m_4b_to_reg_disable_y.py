from mymi.processing.datasets.nifti import create_registered_dataset

dataset = 'LUNG250M-4B'
dest_dataset = 'LUNG250M-4B-REG-NOY'
kwargs = dict(
    disable_y_axis_rotation=True,
)
create_registered_dataset(dataset, dest_dataset, **kwargs)
from mymi.processing.datasets.nifti import create_registered_dataset

dataset = 'LUNG250M-4B'
dest_dataset = 'LUNG250M-4B-REG-MSE-MASK'
kwargs = dict(
    mask_window=(-1024, None),
    metric='mse',
)
create_registered_dataset(dataset, dest_dataset, **kwargs)
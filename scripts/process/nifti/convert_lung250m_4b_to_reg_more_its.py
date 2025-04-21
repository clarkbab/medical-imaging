from mymi.processing.datasets.nifti import create_registered_dataset

dataset = 'LUNG250M-4B'
dest_dataset = 'LUNG250M-4B-REG-MORE-ITS'
create_registered_dataset(dataset, dest_dataset)
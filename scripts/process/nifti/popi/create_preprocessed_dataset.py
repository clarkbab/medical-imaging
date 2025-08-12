from mymi.processing.nifti import convert_to_lung_preprocessed_dataset

dataset = 'POPI'
new_dataset = 'POPI-PP'
kwargs = dict(
    margin=20,
    # pat_ids='pat_0',
)

convert_to_lung_preprocessed_dataset(dataset, new_dataset, **kwargs)
